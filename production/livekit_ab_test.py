"""
Simple live A/B calling test for Deepgram keyword injection.

This script supports two modes:
1) worker: starts a LiveKit telephony agent worker
2) dial: creates an outbound call job (dispatch + SIP participant)

Use this to compare:
- A: Deepgram STT with retrieval-based keyword injection enabled
- B: Deepgram STT with injection disabled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path for src imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from livekit import api
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import deepgram


DEFAULT_LIVEKIT_URL = "wss://healthbot-bfkuu2w0.livekit.cloud"
DEFAULT_LIVEKIT_API_KEY = "APIRGMSsjKnsGuw"
DEFAULT_LIVEKIT_API_SECRET = "P7urIbOH6mknqdhobJ9pthRCfDrp972Xr6KuFlUFz2J"
DEFAULT_AGENT_NAME = "fstt-ab-agent"
DEFAULT_GREETING = "This is Ginko restaurant what would you like to order today"
DEFAULT_TRANSCRIPTS_DIR = "outputs/live_ab_transcripts"
DEFAULT_SIP_TRUNK_ID = "ST_XLdWcK2UnAUx"
DEFAULT_STT_MODEL = "deepgram/nova-3"
DEFAULT_STT_LANGUAGE = "en-US"
DEFAULT_DEEPGRAM_API_KEY = "7970a96b7a1d73b9df6c0ca0a986d995f6866e10"
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_TTS_MODEL = "cartesia/sonic-3"
DEFAULT_CAPTION_OFFSET_SECONDS = 1.8
DEFAULT_PREDICTOR_LOAD_TIMEOUT_SECONDS = 20.0
# Default model dir for preloading (relative to project root, parent of production/)
DEFAULT_MODEL_DIR = "models/retrieval_local_restaurant_type_20260223_181516"
DEFAULT_MAX_TERMS = 50
DEFAULT_TOPK = 50
# Keywords-only (models trained with colab_train_retrieval_local_restaurant_type)
INJECT_MODE = "keywords"
# Max conversation turns to include in keyword forecast input (override via metadata history_turns).
# We forecast terms the user might say next, before they speak.
DEFAULT_HISTORY_TURNS = 4
# Restaurant type for models trained with colab_train_retrieval_local_restaurant_type.
# Must match a type in the training CSV exactly (e.g. chinese_american, thai, greek, ramen).
# The training data has NO plain "chinese"; use "chinese_american" for Chinese cuisine.
DEFAULT_RESTAURANT_TYPE = "asian_fusion" # asian_fusion, chinese
# Seconds to wait after callee joins before greeting (lets audio path establish)
GREETING_DELAY_SECONDS = 1.2

_ENCODER_INDEX_CACHE: dict[str, tuple[Any, Any]] = {}


def _default_model_dir() -> str:
    """Resolve default model path relative to project root."""
    root = Path(__file__).resolve().parent.parent
    return str(root / DEFAULT_MODEL_DIR)


def _apply_default_env() -> None:
    # Let users override these, but make your shared defaults work out-of-the-box.
    os.environ.setdefault("LIVEKIT_URL", DEFAULT_LIVEKIT_URL)
    os.environ.setdefault("LIVEKIT_API_KEY", DEFAULT_LIVEKIT_API_KEY)
    os.environ.setdefault("LIVEKIT_API_SECRET", DEFAULT_LIVEKIT_API_SECRET)
    os.environ.setdefault("LIVEKIT_SIP_TRUNK_ID_OUTBOUND", DEFAULT_SIP_TRUNK_ID)
    # Direct Deepgram mode for keyword injection; use hardcoded key.
    os.environ.setdefault("DEEPGRAM_API_KEY", DEFAULT_DEEPGRAM_API_KEY)
    # Reduce noisy worker telemetry unless explicitly overridden by user.
    os.environ.setdefault("LIVEKIT_LOG_LEVEL", "WARNING")


def _suppress_noisy_logs(record: logging.LogRecord) -> bool:
    """Filter out known noisy LiveKit messages that don't affect behavior."""
    try:
        msg = (record.msg % record.args) if record.args else str(record.msg)
    except (TypeError, ValueError):
        msg = str(record.msg)
    skip = (
        "process memory usage is high" in msg
        or "failed to upload the session report to LiveKit Cloud" in msg
        or "Failed to export logs batch" in msg
        or "project data recording is disabled by owner" in msg
    )
    return not skip


def _configure_logging() -> None:
    level_name = os.getenv("LIVEKIT_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    for name in ("livekit", "livekit.agents", "livekit.api", "opentelemetry.exporter.otlp.proto.http._log_exporter"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addFilter(_suppress_noisy_logs)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return " ".join(p.strip() for p in parts if p and p.strip()).strip()
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text.strip()
    return str(content).strip()


def _normalize_role_for_history(role: str) -> str:
    """Match training format: SYSTEM and USER (assistant → SYSTEM)."""
    r = (role or "").strip().lower()
    if r in {"user", "customer", "human", "client", "guest"}:
        return "USER"
    if r in {"system", "assistant", "agent", "bot", "server"}:
        return "SYSTEM"
    return "SYSTEM"


def _history_text(turns: list[dict[str, str]], max_turns: int = 8) -> str:
    clipped = turns[-max_turns:]
    return "\n".join(
        f"{_normalize_role_for_history(t['role'])}: {t['text']}"
        for t in clipped
        if t["text"].strip()
    )


def _get_audio_duration_seconds(audio_path: Path) -> float | None:
    try:
        import av  # type: ignore

        with av.open(str(audio_path)) as container:
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream and stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


def _load_encoder_index_cached(model_dir: str) -> tuple[Any, Any]:
    """Load encoder + shared_index (FAISS) from src pipeline; cache by model_dir."""
    model_dir_abs = str(Path(model_dir).resolve())
    if model_dir_abs in _ENCODER_INDEX_CACHE:
        return _ENCODER_INDEX_CACHE[model_dir_abs]

    from src.model import load_encoder
    from src.index import VectorIndex

    encoder_path = str(Path(model_dir_abs) / "best_model")
    index_path = str(Path(model_dir_abs) / "shared_index")
    encoder = load_encoder(encoder_path, device="cpu")
    index = VectorIndex.load_shared_index(index_path)
    _ENCODER_INDEX_CACHE[model_dir_abs] = (encoder, index)
    return encoder, index


def _preload_encoder_index(model_dir: str | None) -> None:
    """Load encoder + index at startup so calls have no delay."""
    if not model_dir or not model_dir.strip():
        return
    path = Path(model_dir).resolve()
    if not path.is_dir():
        return
    try:
        _load_encoder_index_cached(str(path))
        print(f"[ab-test] encoder + index preloaded: {path}")
    except Exception as e:
        print(f"[ab-test] encoder+index preload skipped: {e}")


@dataclass
class TranscriptRecorder:
    room_name: str
    call_to: str
    inject_priors: bool
    output_dir: Path
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[dict[str, Any]] = field(default_factory=list)
    history_turns: list[dict[str, str]] = field(default_factory=list)
    injected_terms_log: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)

    def add_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append(
            {
                "time": _utc_now(),
                "type": event_type,
                "payload": payload,
            }
        )

    def add_turn(self, role: str, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        self.history_turns.append({"role": role, "text": clean})
        self.add_event("turn", {"role": role, "text": clean})

    def log_terms(self, terms: list[str], history_snapshot: str) -> None:
        self.injected_terms_log.append(
            {
                "time": _utc_now(),
                "terms": terms,
                "history_snapshot": history_snapshot,
            }
        )

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f"__{self.room_name}__{'inject_on' if self.inject_priors else 'inject_off'}.json"
        )
        path = self.output_dir / filename
        payload = {
            "session_id": self.session_id,
            "room_name": self.room_name,
            "call_to": self.call_to,
            "inject_priors": self.inject_priors,
            "started_at": self.started_at,
            "ended_at": _utc_now(),
            "history_turns": self.history_turns,
            "events": self.events,
            "injected_terms_log": self.injected_terms_log,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _build_segments(
        self,
        caption_offset_seconds: float = DEFAULT_CAPTION_OFFSET_SECONDS,
        audio_duration_seconds: float | None = None,
    ) -> list[tuple[float, float, str]]:
        turn_events = [e for e in self.events if e.get("type") == "turn"]
        if not turn_events:
            return []

        base_dt = datetime.fromisoformat(turn_events[0]["time"])
        raw: list[tuple[float, float, str]] = []

        for idx, ev in enumerate(turn_events):
            payload = ev.get("payload", {})
            role = str(payload.get("role", "unknown")).upper()
            text = str(payload.get("text", "")).strip()
            if not text:
                continue

            start_dt = datetime.fromisoformat(ev["time"])
            if idx + 1 < len(turn_events):
                end_dt = datetime.fromisoformat(turn_events[idx + 1]["time"])
            else:
                end_dt = start_dt + timedelta(seconds=4)

            start_s = (start_dt - base_dt).total_seconds() + caption_offset_seconds
            end_s = (end_dt - base_dt).total_seconds() + caption_offset_seconds
            if end_s <= start_s + 0.2:
                end_s = start_s + 2.0

            raw.append((start_s, end_s, f"{role}: {text}"))

        # Keep starts non-negative and ordered.
        segments: list[tuple[float, float, str]] = []
        last_end = 0.0
        for start_s, end_s, text in raw:
            start_s = max(start_s, 0.0)
            end_s = max(end_s, start_s + 0.6)
            if start_s < last_end:
                shift = last_end - start_s
                start_s += shift
                end_s += shift
            segments.append((start_s, end_s, text))
            last_end = end_s

        # If we know audio duration, gently scale timeline so final subtitle aligns.
        if audio_duration_seconds and audio_duration_seconds > 1.0 and segments:
            last_end = segments[-1][1]
            target_end = max(audio_duration_seconds - 0.2, 0.5)
            if last_end > 0.1:
                ratio = target_end / last_end
                if 0.7 <= ratio <= 1.3:
                    scaled: list[tuple[float, float, str]] = []
                    for start_s, end_s, text in segments:
                        s = max(start_s * ratio, 0.0)
                        e = max(end_s * ratio, s + 0.6)
                        scaled.append((s, e, text))
                    segments = scaled

        return segments

    def write_vtt(
        self,
        json_path: Path,
        caption_offset_seconds: float = DEFAULT_CAPTION_OFFSET_SECONDS,
        audio_duration_seconds: float | None = None,
    ) -> Path | None:
        def _fmt_vtt_time(seconds: float) -> str:
            seconds = max(seconds, 0.0)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

        segments = self._build_segments(caption_offset_seconds, audio_duration_seconds)
        if not segments:
            return None

        cues: list[str] = ["WEBVTT", ""]
        for start_s, end_s, text in segments:
            cues.append(f"{_fmt_vtt_time(start_s)} --> {_fmt_vtt_time(end_s)}")
            cues.append(text)
            cues.append("")

        vtt_path = json_path.with_suffix(".vtt")
        vtt_path.write_text("\n".join(cues), encoding="utf-8")
        return vtt_path

    def write_srt(
        self,
        json_path: Path,
        caption_offset_seconds: float = DEFAULT_CAPTION_OFFSET_SECONDS,
        audio_duration_seconds: float | None = None,
    ) -> Path | None:
        def _fmt_srt_time(seconds: float) -> str:
            seconds = max(seconds, 0.0)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        segments = self._build_segments(caption_offset_seconds, audio_duration_seconds)
        if not segments:
            return None

        cues: list[str] = []
        cue_idx = 1
        for start_s, end_s, text in segments:
            cues.append(str(cue_idx))
            cues.append(f"{_fmt_srt_time(start_s)} --> {_fmt_srt_time(end_s)}")
            cues.append(text)
            cues.append("")
            cue_idx += 1

        srt_path = json_path.with_suffix(".srt")
        srt_path.write_text("\n".join(cues), encoding="utf-8")
        return srt_path

    def write_player_html(
        self,
        json_path: Path,
        audio_path: Path,
        vtt_path: Path | None,
        caption_offset_seconds: float = DEFAULT_CAPTION_OFFSET_SECONDS,
        audio_duration_seconds: float | None = None,
    ) -> Path:
        segments = self._build_segments(caption_offset_seconds, audio_duration_seconds)
        cues_payload = [{"start": s, "end": e, "text": t} for s, e, t in segments]
        cues_json = json.dumps(cues_payload, ensure_ascii=False)
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Call Playback</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 20px;
    }}
    #captions {{
      margin-top: 12px;
      min-height: 48px;
      padding: 10px 12px;
      border-radius: 8px;
      background: #111;
      color: #fff;
      font-size: 15px;
      line-height: 1.4;
      white-space: pre-wrap;
    }}
    #meta {{
      color: #666;
      font-size: 12px;
      margin-top: 6px;
    }}
    #controls {{
      margin-top: 10px;
      font-size: 13px;
      color: #333;
    }}
    #transcript {{
      margin-top: 14px;
      border: 1px solid #ddd;
      border-radius: 8px;
      max-height: 320px;
      overflow: auto;
    }}
    .line {{
      padding: 8px 10px;
      border-bottom: 1px solid #f2f2f2;
      font-size: 14px;
      line-height: 1.35;
    }}
    .line.active {{
      background: #fff4cc;
    }}
  </style>
</head>
<body>
  <h3>{json_path.stem}</h3>
  <audio id="player" controls style="width: 100%;">
    <source src="{audio_path.name}" type="audio/ogg">
  </audio>
  <div id="captions">(captions will appear during playback)</div>
  <div id="controls">
    Caption offset: <input id="offset" type="range" min="-8" max="8" step="0.1" value="0"> <span id="offsetVal">0.0s</span>
  </div>
  <div id="meta">Captions are synced from saved transcript events. Increase offset if text appears early.</div>
  <div id="transcript"></div>

  <script>
    const audio = document.getElementById("player");
    const captionBox = document.getElementById("captions");
    const transcript = document.getElementById("transcript");
    const offsetInput = document.getElementById("offset");
    const offsetVal = document.getElementById("offsetVal");
    const CUES = {cues_json};
    let activeIndex = -1;

    function setCaption(text) {{
      captionBox.textContent = text && text.trim() ? text : "";
    }}

    function renderTranscript() {{
      transcript.innerHTML = "";
      for (let i = 0; i < CUES.length; i += 1) {{
        const row = document.createElement("div");
        row.className = "line";
        row.dataset.idx = String(i);
        row.textContent = CUES[i].text;
        row.addEventListener("click", () => {{
          audio.currentTime = Math.max(CUES[i].start - parseFloat(offsetInput.value || "0"), 0);
          syncCaption();
        }});
        transcript.appendChild(row);
      }}
    }}

    function syncCaption() {{
      if (!CUES || CUES.length === 0) {{
        setCaption("(no saved captions)");
        return;
      }}
      const t = (audio.currentTime || 0) + parseFloat(offsetInput.value || "0");
      let active = "";
      let nextIdx = -1;
      for (let i = 0; i < CUES.length; i += 1) {{
        const cue = CUES[i];
        if (t >= cue.start && t < cue.end) {{
          active = cue.text;
          nextIdx = i;
          break;
        }}
      }}
      setCaption(active);
      if (nextIdx !== activeIndex) {{
        if (activeIndex >= 0) {{
          const prev = transcript.querySelector(`.line[data-idx="${{activeIndex}}"]`);
          if (prev) prev.classList.remove("active");
        }}
        if (nextIdx >= 0) {{
          const curr = transcript.querySelector(`.line[data-idx="${{nextIdx}}"]`);
          if (curr) {{
            curr.classList.add("active");
            curr.scrollIntoView({{ block: "nearest" }});
          }}
        }}
        activeIndex = nextIdx;
      }}
    }}

    audio.addEventListener("timeupdate", syncCaption);
    audio.addEventListener("seeked", syncCaption);
    audio.addEventListener("loadedmetadata", syncCaption);
    offsetInput.addEventListener("input", () => {{
      offsetVal.textContent = `${{parseFloat(offsetInput.value).toFixed(1)}}s`;
      syncCaption();
    }});
    window.addEventListener("load", () => {{
      renderTranscript();
      syncCaption();
    }});
  </script>
</body>
</html>
"""
        html_path = json_path.with_suffix(".player.html")
        html_path.write_text(html, encoding="utf-8")
        return html_path

    def write_vlc_playlist(self, audio_path: Path, srt_path: Path) -> Path:
        m3u = "\n".join(
            [
                "#EXTM3U",
                f"#EXTVLCOPT:sub-file={srt_path.name}",
                audio_path.name,
                "",
            ]
        )
        playlist_path = audio_path.with_suffix(".vlc.m3u")
        playlist_path.write_text(m3u, encoding="utf-8")
        return playlist_path


def _build_agent_instructions() -> str:
    return (
        "You are taking phone orders for a Chinese restaurant. "
        "Keep replies short and natural for a phone call. "
        "Ask clarifying questions only when needed. "
        "Confirm the order details before ending."
    )


def _build_stt(use_injection: bool):
    if use_injection:
        return deepgram.STT(
            model="nova-3",
            language=os.getenv("LK_STT_LANGUAGE", DEFAULT_STT_LANGUAGE),
            punctuate=True,
            smart_format=True,
            interim_results=True,
            keyterm=[],
        )
    return os.getenv("LK_STT_MODEL", DEFAULT_STT_MODEL)


async def _worker_entrypoint(ctx: JobContext) -> None:
    metadata_raw = getattr(ctx.job, "metadata", "") or "{}"
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        metadata = {}

    inject_priors = bool(metadata.get("inject_priors", False))
    call_to = str(metadata.get("call_to", "unknown"))
    output_dir = Path(str(metadata.get("output_dir", DEFAULT_TRANSCRIPTS_DIR)))
    greeting = str(metadata.get("greeting", DEFAULT_GREETING))
    model_dir = metadata.get("model_dir")
    max_terms = int(metadata.get("max_terms", DEFAULT_MAX_TERMS))
    topk = int(metadata.get("topk", DEFAULT_TOPK))
    history_turns = int(metadata.get("history_turns", DEFAULT_HISTORY_TURNS))
    restaurant_type = metadata.get("restaurant_type", DEFAULT_RESTAURANT_TYPE)
    if isinstance(restaurant_type, str):
        restaurant_type = restaurant_type.strip() or None
    sip_identity = str(metadata.get("sip_participant_identity", "")).strip() or None
    max_call_seconds = int(metadata.get("max_call_seconds", 900))
    caption_offset_seconds = float(
        metadata.get("caption_offset_seconds", DEFAULT_CAPTION_OFFSET_SECONDS)
    )

    recorder = TranscriptRecorder(
        room_name=ctx.room.name,
        call_to=call_to,
        inject_priors=inject_priors,
        output_dir=output_dir,
    )

    encoder, index = None, None
    if inject_priors:
        if not model_dir:
            recorder.add_event(
                "warning",
                {"message": "inject_priors=true but model_dir missing; disabling injection"},
            )
            inject_priors = False
        else:
            timeout_s = float(
                metadata.get("predictor_load_timeout_seconds", DEFAULT_PREDICTOR_LOAD_TIMEOUT_SECONDS)
            )
            try:
                encoder, index = await asyncio.wait_for(
                    asyncio.to_thread(_load_encoder_index_cached, str(model_dir)),
                    timeout=timeout_s,
                )
            except Exception as exc:
                recorder.add_event(
                    "warning",
                    {
                        "message": "encoder+index load failed or timed out; disabling injection for this call",
                        "error": str(exc),
                    },
                )
                inject_priors = False

    stt = _build_stt(inject_priors)
    session = AgentSession(
        stt=stt,
        llm=os.getenv("LK_DIALOG_LLM_MODEL", DEFAULT_LLM_MODEL),
        tts=os.getenv("LK_TTS_MODEL", DEFAULT_TTS_MODEL),
        max_endpointing_delay=2.0,
    )
    agent = Agent(instructions=_build_agent_instructions())

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event) -> None:
        recorder.add_event(
            "user_input_transcribed",
            {
                "transcript": event.transcript,
                "is_final": bool(event.is_final),
                "language": event.language,
                "speaker_id": event.speaker_id,
            },
        )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event) -> None:
        item = event.item
        role_raw = str(getattr(item, "role", "unknown"))
        role = _normalize_role_for_history(role_raw)  # assistant → SYSTEM, user → USER
        text = _coerce_text(getattr(item, "content", None))
        recorder.add_turn(role, text)

        # Only forecast keywords when SYSTEM spoke (user's next turn is coming).
        # We don't know what the user will say yet—we forecast terms to bias transcription.
        # Forecasting on USER turn would use history without the system's reply yet.
        if role != "SYSTEM":
            return

        if not inject_priors or encoder is None or index is None:
            return

        history = _history_text(recorder.history_turns, max_turns=history_turns)
        if not history.strip():
            return
        # Prepend restaurant type for models trained with restaurant_type input
        if restaurant_type:
            history = f"Restaurant type: {restaurant_type}\n\n{history}"
        try:
            from src.livekit_deepgram_inject import predict_terms, extract_explicit_options_from_utterance

            terms = predict_terms(
                encoder, index, history,
                topk=topk, max_terms=max_terms,
                inject_mode=INJECT_MODE,
            )
            # Prepend explicit options from the agent's last utterance (e.g. "chicken pork
            # shrimp" when agent said "we have chicken pork shrimp and vegetable lo mein").
            # Retrieval can miss these; extracting them ensures they are always injected.
            explicit = extract_explicit_options_from_utterance(text)
            if explicit:
                explicit_set = {t.lower() for t in explicit}
                terms = explicit + [t for t in terms if t.lower() not in explicit_set]
            terms = terms[:max_terms]
        except Exception as exc:
            recorder.add_event("prior_prediction_error", {"error": str(exc)})
            return

        if hasattr(stt, "update_options"):
            stt.update_options(keyterm=terms)
        recorder.log_terms(terms, history)
        recorder.add_event("deepgram_keyword_update", {"count": len(terms), "terms": terms})

    await session.start(
        agent=agent,
        room=ctx.room,
        record={"audio": True, "logs": False, "traces": False, "transcript": False},
    )

    # Wait for the PSTN callee to appear.
    if sip_identity:
        await ctx.wait_for_participant(identity=sip_identity)
    else:
        await ctx.wait_for_participant()

    # Brief delay so audio path (LiveKit -> SIP -> PSTN) is established before greeting.
    await asyncio.sleep(GREETING_DELAY_SECONDS)

    greeting_handle = session.say(greeting, add_to_chat_ctx=True)
    await greeting_handle

    disconnected = asyncio.Event()

    def _on_participant_disconnected(participant) -> None:
        if sip_identity and participant.identity != sip_identity:
            return
        disconnected.set()

    ctx.room.on("participant_disconnected", _on_participant_disconnected)

    try:
        await asyncio.wait_for(disconnected.wait(), timeout=max_call_seconds)
    except asyncio.TimeoutError:
        recorder.add_event(
            "timeout",
            {"message": f"call exceeded {max_call_seconds}s, closing session"},
        )
    finally:
        await session.drain()
        await session.aclose()
        out_path = recorder.save()

        session_audio = Path(ctx.session_directory) / "audio.ogg"
        audio_out_path: Path | None = None
        audio_duration_seconds: float | None = None
        if session_audio.is_file():
            audio_out_path = out_path.with_suffix(".ogg")
            shutil.copy2(session_audio, audio_out_path)
            audio_duration_seconds = _get_audio_duration_seconds(audio_out_path)

        vtt_path = recorder.write_vtt(
            out_path,
            caption_offset_seconds=caption_offset_seconds,
            audio_duration_seconds=audio_duration_seconds,
        )
        srt_path = recorder.write_srt(
            out_path,
            caption_offset_seconds=caption_offset_seconds,
            audio_duration_seconds=audio_duration_seconds,
        )

        player_path: Path | None = None
        vlc_playlist_path: Path | None = None
        if audio_out_path is not None:
            player_path = recorder.write_player_html(
                out_path,
                audio_out_path,
                vtt_path,
                caption_offset_seconds=caption_offset_seconds,
                audio_duration_seconds=audio_duration_seconds,
            )
            if srt_path:
                vlc_playlist_path = recorder.write_vlc_playlist(audio_out_path, srt_path)

        print(f"[ab-test] transcript saved: {out_path}")
        if audio_out_path:
            print(f"[ab-test] audio saved: {audio_out_path}")
        if vtt_path:
            print(f"[ab-test] captions saved: {vtt_path}")
        if srt_path:
            print(f"[ab-test] srt saved: {srt_path}")
        if player_path:
            print(f"[ab-test] player saved: {player_path}")
        if vlc_playlist_path:
            print(f"[ab-test] vlc playlist saved: {vlc_playlist_path}")


def _make_server() -> AgentServer:
    server = AgentServer()
    server.rtc_session(_worker_entrypoint, agent_name=DEFAULT_AGENT_NAME)
    return server


def _build_metadata(args: argparse.Namespace, sip_participant_identity: str) -> str:
    payload = {
        "call_to": args.call_to,
        "inject_priors": bool(args.inject_priors),
        "model_dir": args.model_dir,
        "max_terms": args.max_terms,
        "topk": args.topk,
        "history_turns": getattr(args, "history_turns", DEFAULT_HISTORY_TURNS),
        "restaurant_type": getattr(args, "restaurant_type", None) or None,
        "output_dir": args.output_dir,
        "greeting": args.greeting,
        "sip_participant_identity": sip_participant_identity,
        "max_call_seconds": args.max_call_seconds,
        "caption_offset_seconds": args.caption_offset_seconds,
        "predictor_load_timeout_seconds": args.predictor_load_timeout_seconds,
    }
    return json.dumps(payload)


async def _dial(args: argparse.Namespace) -> None:
    room_name = f"ab-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    sip_participant_identity = f"sip-callee-{uuid.uuid4().hex[:8]}"

    metadata = _build_metadata(args, sip_participant_identity=sip_participant_identity)
    lkapi = api.LiveKitAPI()
    try:
        await lkapi.room.create_room(api.CreateRoomRequest(name=room_name))

        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name=args.agent_name,
                metadata=metadata,
            )
        )

        sip_participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=room_name,
                participant_identity=sip_participant_identity,
                participant_name="Customer",
                sip_trunk_id=args.sip_trunk_id,
                sip_call_to=args.call_to,
                wait_until_answered=True,
            )
        )

        print(
            json.dumps(
                {
                    "status": "call_started",
                    "room_name": room_name,
                    "dispatch_id": dispatch.id,
                    "sip_participant_identity": sip_participant_identity,
                    "sip_call_id": getattr(sip_participant, "sip_call_id", ""),
                    "inject_priors": bool(args.inject_priors),
                    "output_dir": args.output_dir,
                },
                indent=2,
            )
        )
    finally:
        close_fn = getattr(lkapi, "aclose", None)
        if callable(close_fn):
            await close_fn()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiveKit + Deepgram A/B phone call test")
    sub = parser.add_subparsers(dest="mode", required=True)

    worker_parser = sub.add_parser("worker", help="Run the LiveKit agent worker")
    worker_parser.add_argument(
        "--model-dir",
        default=os.getenv("FSTT_MODEL_DIR", _default_model_dir()),
        help="Preload predictor + FAISS index at startup (no per-call delay)",
    )

    dial = sub.add_parser("dial", help="Start one outbound A/B call")
    dial.add_argument("--call-to", required=True, help="Target phone number in E.164 format")
    dial.add_argument(
        "--sip-trunk-id",
        default=os.getenv("LIVEKIT_SIP_TRUNK_ID_OUTBOUND", DEFAULT_SIP_TRUNK_ID),
        help="LiveKit outbound SIP trunk ID",
    )
    dial.add_argument(
        "--agent-name",
        default=os.getenv("AB_AGENT_NAME", DEFAULT_AGENT_NAME),
        help="Explicit LiveKit agent name used for dispatch",
    )
    dial.add_argument(
        "--inject-priors",
        action="store_true",
        help="Enable retrieval-based keyword injection",
    )
    dial.add_argument(
        "--model-dir",
        default=os.getenv("FSTT_MODEL_DIR", _default_model_dir()),
        help="Model dir for injection (optional; defaults to same path preloaded by worker)",
    )
    dial.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS, help="Max keywords to inject")
    dial.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Top-k retrieval depth")
    dial.add_argument(
        "--history-turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        help="Max conversation turns for keyword forecast input",
    )
    dial.add_argument(
        "--restaurant-type",
        type=str,
        default=None,
        help="Restaurant type for models trained with restaurant_type. Must match CSV exactly "
        "(e.g. chinese_american, thai, greek, ramen, vietnamese, korean).",
    )
    dial.add_argument(
        "--output-dir",
        default=DEFAULT_TRANSCRIPTS_DIR,
        help="Directory where transcript JSON files are written",
    )
    dial.add_argument(
        "--greeting",
        default=DEFAULT_GREETING,
        help="First sentence spoken after callee answers",
    )
    dial.add_argument(
        "--max-call-seconds",
        type=int,
        default=900,
        help="Hard timeout for the call session",
    )
    dial.add_argument(
        "--caption-offset-seconds",
        type=float,
        default=DEFAULT_CAPTION_OFFSET_SECONDS,
        help="Shift captions later (+) or earlier (-) to align with audio",
    )
    dial.add_argument(
        "--predictor-load-timeout-seconds",
        type=float,
        default=DEFAULT_PREDICTOR_LOAD_TIMEOUT_SECONDS,
        help="Timeout for loading retrieval predictor before fallback to no-injection",
    )
    return parser


def main() -> None:
    _apply_default_env()
    _configure_logging()
    args = _parser().parse_args()

    if args.mode == "worker":
        _preload_encoder_index(getattr(args, "model_dir", None))
        server = _make_server()
        # LiveKit's runner expects a command like "start"; this script already
        # chose mode via argparse, so force the worker runner command here.
        sys.argv = [sys.argv[0], "start"]
        cli.run_app(server)
        return

    if args.mode == "dial":
        if not args.sip_trunk_id:
            raise ValueError(
                "Missing --sip-trunk-id (or LIVEKIT_SIP_TRUNK_ID_OUTBOUND env var)."
            )
        asyncio.run(_dial(args))
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
