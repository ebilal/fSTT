import argparse
import csv
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.eval_shokudo import load_index_with_fallback, resolve_encoder_dir, resolve_index_dir
from src.model import encode_texts, load_encoder
from src.shokudo_eval import prepare_predictions
from src.utils import ensure_dir, get_device

PART_RE = re.compile(r"^(?P<base>.+)_part(?P<part>[12])\.mp3$", re.IGNORECASE)

# Deepgram supports up to 100 keyterms.
DEEPGRAM_MAX_KEYTERMS = 100

# Common English stopwords that provide no biasing value for STT.
_STOPWORDS = frozenset(
    "a about above after again against all am an and any are aren't as at be because been before being below "
    "between both but by can could couldn't did didn't do does doesn't doing don don't down during each few for "
    "from further get got great had hadn't has hasn't have haven't having he her here hers herself him himself "
    "his how i if in into is isn't it its itself just let like ll m me might more most mustn't my myself no nor "
    "not now of off oh ok okay on once only or other our ours ourselves out over own pls please re s same shall "
    "shan't she should shouldn't so some sounds such sure t than thank thanks that that'll the their theirs them "
    "themselves then there these they this those through to too under until up us ve very want was wasn't we well "
    "were weren't what when where which while who whom why will with won won't would wouldn't ya yeah yes yet "
    "you your yours yourself yourselves yay nope think believe need come comes came nice good going go going "
    "got know let's look looks make makes much really right s2 see take tell thing things thought try um uh".split()
)


def _filter_stopwords(terms: List[str]) -> List[str]:
    """Remove terms that are pure stopwords (single-word) or where every word is a stopword (multi-word)."""
    filtered: List[str] = []
    for term in terms:
        words = term.lower().split()
        if not words:
            continue
        # Keep if at least one word is NOT a stopword
        if any(w not in _STOPWORDS for w in words):
            filtered.append(term)
    return filtered


def _interleave(a: List[str], b: List[str]) -> List[str]:
    """Interleave two lists so both are fairly represented, deduplicating by lowercase."""
    result: List[str] = []
    seen: set = set()
    i, j = 0, 0
    while i < len(a) or j < len(b):
        if i < len(a):
            key = a[i].lower()
            if key not in seen:
                seen.add(key)
                result.append(a[i])
            i += 1
        if j < len(b):
            key = b[j].lower()
            if key not in seen:
                seen.add(key)
                result.append(b[j])
            j += 1
    return result


def discover_pairs(split_dir: str) -> List[Tuple[str, str, str]]:
    grouped: Dict[str, Dict[str, str]] = {}
    for name in sorted(os.listdir(split_dir)):
        if not name.lower().endswith(".mp3"):
            continue
        match = PART_RE.match(name)
        if not match:
            continue
        base = match.group("base")
        part = match.group("part")
        grouped.setdefault(base, {})[part] = os.path.join(split_dir, name)

    pairs: List[Tuple[str, str, str]] = []
    for base in sorted(grouped.keys()):
        parts = grouped[base]
        if "1" in parts and "2" in parts:
            pairs.append((base, parts["1"], parts["2"]))
    return pairs


def resolve_prompt_terms(
    model: str,
    keywords: List[str],
    keyterms: List[str],
    max_injected_terms: int,
) -> Dict[str, List[str]]:
    # Hard-cap to Deepgram's documented limit.
    effective_cap = min(max_injected_terms, DEEPGRAM_MAX_KEYTERMS)
    is_nova3 = "nova-3" in (model or "").lower()

    # Clean inputs.
    clean_kw = _filter_stopwords([(t or "").strip() for t in keywords if (t or "").strip()])
    clean_kt = _filter_stopwords([(t or "").strip() for t in keyterms if (t or "").strip()])

    if is_nova3:
        # Nova-3 only supports keyterm prompting — interleave keywords and
        # keyterms so both types are represented after truncation.
        merged = _interleave(clean_kw, clean_kt)
        return {"keywords": [], "keyterms": merged[:effective_cap]}
    return {
        "keywords": clean_kw[:effective_cap],
        "keyterms": clean_kt[:effective_cap],
    }


def deepgram_transcribe(
    audio_path: str,
    api_key: str,
    model: str,
    language: str,
    keywords: List[str],
    keyterms: List[str],
    max_injected_terms: int,
    diarize: bool = False,
    utterances: bool = False,
) -> Dict[str, object]:
    is_nova3 = "nova-3" in (model or "").lower()
    params: List[Tuple[str, str]] = [
        ("model", model),
        ("smart_format", "true"),
        ("punctuate", "true"),
    ]
    if language:
        params.append(("language", language))
    if diarize:
        params.append(("diarize", "true"))
    if utterances:
        params.append(("utterances", "true"))

    prompt_terms = resolve_prompt_terms(model, keywords, keyterms, max_injected_terms=max_injected_terms)
    if is_nova3:
        for term in prompt_terms["keyterms"]:
            params.append(("keyterm", term))
    else:
        for kw in prompt_terms["keywords"]:
            params.append(("keywords", kw))
        for kt in prompt_terms["keyterms"]:
            params.append(("keyterm", kt))

    query = urllib.parse.urlencode(params, doseq=True)
    url = f"https://api.deepgram.com/v1/listen?{query}"

    content_type = mimetypes.guess_type(audio_path)[0] or "audio/mpeg"
    with open(audio_path, "rb") as f:
        payload = f.read()

    req = urllib.request.Request(
        url=url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": content_type,
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def extract_transcript(deepgram_response: Dict[str, object]) -> str:
    try:
        return (
            deepgram_response["results"]["channels"][0]["alternatives"][0]["transcript"]  # type: ignore[index]
            or ""
        ).strip()
    except Exception:
        return ""


def build_history_from_utterances(deepgram_response: Dict[str, object]) -> str:
    utterances = deepgram_response.get("results", {}).get("utterances", [])  # type: ignore[assignment]
    if not isinstance(utterances, list) or not utterances:
        return ""

    # Map first detected speaker to SYSTEM to match training format.
    first_speaker = utterances[0].get("speaker")
    speaker_map: Dict[object, str] = {first_speaker: "SYSTEM"}
    role_toggle = "USER"

    lines: List[str] = []
    for utt in utterances:
        transcript = (utt.get("transcript") or "").strip()
        if not transcript:
            continue
        speaker_id = utt.get("speaker")
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = role_toggle
            role_toggle = "SYSTEM" if role_toggle == "USER" else "USER"
        role = speaker_map.get(speaker_id, "USER")
        lines.append(f"{role}: {transcript}")
    return "\n".join(lines).strip()


def clean_deepgram_terms(terms: List[str]) -> List[str]:
    cleaned: List[str] = []
    for term in terms:
        text = (term or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Deepgram Nova-3 with and without retrieval priors on split calls.")
    parser.add_argument("--audio_split_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--deepgram_api_key", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/deepgram_split_priors")
    parser.add_argument("--model", type=str, default="nova-3")
    parser.add_argument("--language", type=str, default="")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max_keywords", type=int, default=30)
    parser.add_argument("--max_keyterms", type=int, default=30)
    parser.add_argument("--max_injected_terms", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = get_device(args.device)

    print(f"Loading model/index from {args.run_dir}")
    encoder_dir = resolve_encoder_dir(args.run_dir)
    index_dir = resolve_index_dir(args.run_dir)
    encoder = load_encoder(encoder_dir, device=device)
    index = load_index_with_fallback(index_dir, encoder)
    print(f"Encoder: {encoder_dir}")
    print(f"Index:   {index_dir}")

    pairs = discover_pairs(args.audio_split_dir)
    if not pairs:
        raise ValueError(f"No part1/part2 pairs found in {args.audio_split_dir}")
    print(f"Found {len(pairs)} conversation pairs")

    rows: List[Dict[str, str]] = []
    json_records: List[Dict[str, object]] = []

    for idx, (conv_id, part1_path, part2_path) in enumerate(pairs, start=1):
        print(f"\n[{idx}/{len(pairs)}] {conv_id}")
        pair_start = time.perf_counter()

        # 1) Transcribe part1 (no bias), then predict priors from retrieval model.
        part1_resp = deepgram_transcribe(
            audio_path=part1_path,
            api_key=args.deepgram_api_key,
            model=args.model,
            language=args.language,
            keywords=[],
            keyterms=[],
            max_injected_terms=args.max_injected_terms,
            diarize=True,
            utterances=True,
        )
        part1_text = extract_transcript(part1_resp)
        history_text = build_history_from_utterances(part1_resp)
        if not history_text and part1_text:
            # Fallback if utterance diarization is unavailable.
            history_text = f"USER: {part1_text}"
        history_emb = encode_texts(encoder, [history_text])
        retrieved = index.retrieve_texts(history_emb, args.topk)[0]
        pred = prepare_predictions(
            retrieved,
            max_keywords=args.max_keywords,
            max_keyterms=args.max_keyterms,
        )
        # Strip empty padding from prepare_predictions; keep original case.
        pred_keywords = [t.strip() for t in pred["keywords"] if t and t.strip()]
        pred_keyterms = [t.strip() for t in pred["keyterms"] if t and t.strip()]

        # 2) Transcribe part2 in four modes.
        variants = [
            ("no_bias", [], []),
            ("keywords_only", pred_keywords, []),
            ("keyterms_only", [], pred_keyterms),
            ("keywords_plus_keyterms", pred_keywords, pred_keyterms),
        ]

        outputs: Dict[str, str] = {}
        sent_terms_by_variant: Dict[str, Dict[str, List[str]]] = {}
        for variant_name, kws, kts in variants:
            sent_terms = resolve_prompt_terms(
                args.model,
                kws,
                kts,
                max_injected_terms=args.max_injected_terms,
            )
            sent_terms_by_variant[variant_name] = sent_terms
            try:
                resp = deepgram_transcribe(
                    audio_path=part2_path,
                    api_key=args.deepgram_api_key,
                    model=args.model,
                    language=args.language,
                    keywords=kws,
                    keyterms=kts,
                    max_injected_terms=args.max_injected_terms,
                )
                outputs[variant_name] = extract_transcript(resp)
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="ignore")
                outputs[variant_name] = f"[HTTPError {e.code}] {err_text[:500]}"
            except Exception as e:
                outputs[variant_name] = f"[Error] {e}"

        rows.append(
            {
                "conversation_id": conv_id,
                "part1_file": os.path.basename(part1_path),
                "part2_file": os.path.basename(part2_path),
                "part1_transcript": part1_text,
                "part1_history_for_model": history_text,
                "pred_keywords_30": json.dumps(pred_keywords, ensure_ascii=False),
                "pred_keyterms_30": json.dumps(pred_keyterms, ensure_ascii=False),
                "retrieved_top5": json.dumps(retrieved[:5], ensure_ascii=False),
                "part2_no_bias": outputs.get("no_bias", ""),
                "part2_keywords_only": outputs.get("keywords_only", ""),
                "part2_keyterms_only": outputs.get("keyterms_only", ""),
                "part2_keywords_plus_keyterms": outputs.get("keywords_plus_keyterms", ""),
                "sent_keywords_keywords_only": json.dumps(
                    sent_terms_by_variant.get("keywords_only", {}).get("keywords", []),
                    ensure_ascii=False,
                ),
                "sent_keywords_keyterms_only": json.dumps(
                    sent_terms_by_variant.get("keyterms_only", {}).get("keywords", []),
                    ensure_ascii=False,
                ),
                "sent_keywords_keywords_plus_keyterms": json.dumps(
                    sent_terms_by_variant.get("keywords_plus_keyterms", {}).get("keywords", []),
                    ensure_ascii=False,
                ),
                "sent_keyterms_keywords_only": json.dumps(
                    sent_terms_by_variant.get("keywords_only", {}).get("keyterms", []),
                    ensure_ascii=False,
                ),
                "sent_keyterms_keyterms_only": json.dumps(
                    sent_terms_by_variant.get("keyterms_only", {}).get("keyterms", []),
                    ensure_ascii=False,
                ),
                "sent_keyterms_keywords_plus_keyterms": json.dumps(
                    sent_terms_by_variant.get("keywords_plus_keyterms", {}).get("keyterms", []),
                    ensure_ascii=False,
                ),
            }
        )

        json_records.append(
            {
                "conversation_id": conv_id,
                "part1_file": part1_path,
                "part2_file": part2_path,
                "part1_transcript": part1_text,
                "part1_history_for_model": history_text,
                "pred_keywords_30": pred_keywords,
                "pred_keyterms_30": pred_keyterms,
                "retrieved_top5": retrieved[:5],
                "sent_terms_by_variant": sent_terms_by_variant,
                "part2_outputs": outputs,
            }
        )

        print(
            f"part1 chars={len(part1_text)} | kw={len(pred_keywords)} | kt={len(pred_keyterms)} | "
            f"elapsed={time.perf_counter() - pair_start:.1f}s"
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"deepgram_split_priors_compare_{stamp}.csv")
    json_path = os.path.join(args.out_dir, f"deepgram_split_priors_compare_{stamp}.json")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_id",
                "part1_file",
                "part2_file",
                "part1_transcript",
                "part1_history_for_model",
                "pred_keywords_30",
                "pred_keyterms_30",
                "retrieved_top5",
                "part2_no_bias",
                "part2_keywords_only",
                "part2_keyterms_only",
                "part2_keywords_plus_keyterms",
                "sent_keywords_keywords_only",
                "sent_keywords_keyterms_only",
                "sent_keywords_keywords_plus_keyterms",
                "sent_keyterms_keywords_only",
                "sent_keyterms_keyterms_only",
                "sent_keyterms_keywords_plus_keyterms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": args.run_dir,
                "encoder_dir": encoder_dir,
                "index_dir": index_dir,
                "deepgram_model": args.model,
                "language": args.language,
                "topk": args.topk,
                "max_keywords": args.max_keywords,
                "max_keyterms": args.max_keyterms,
                "max_injected_terms": args.max_injected_terms,
                "num_pairs": len(json_records),
                "records": json_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDone. CSV: {csv_path}")
    print(f"Done. JSON: {json_path}")


if __name__ == "__main__":
    main()
