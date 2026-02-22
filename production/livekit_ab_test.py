"""
Simple live A/B calling test for Deepgram keyterm injection.

This script supports two modes:
1) worker: starts a LiveKit telephony agent worker
2) dial: creates an outbound call job (dispatch + SIP participant)

Use this to compare:
- A: Deepgram STT with retrieval-based keyterm injection enabled
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

from livekit import api
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import deepgram


DEFAULT_LIVEKIT_URL = "wss://healthbot-bfkuu2w0.livekit.cloud"
DEFAULT_LIVEKIT_API_KEY = "APIRGMSsjKnsGuw"
DEFAULT_LIVEKIT_API_SECRET = "P7urIbOH6mknqdhobJ9pthRCfDrp972Xr6KuFlUFz2J"
DEFAULT_AGENT_NAME = "fstt-ab-agent"
DEFAULT_GREETING = "This is chinese restaurant what would you like to order today"
DEFAULT_TRANSCRIPTS_DIR = "outputs/live_ab_transcripts"
DEFAULT_SIP_TRUNK_ID = "ST_XLdWcK2UnAUx"
DEFAULT_STT_MODEL = "deepgram/nova-3"
DEFAULT_STT_LANGUAGE = "en-US"
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_TTS_MODEL = "cartesia/sonic-3"


def _apply_default_env() -> None:
    # Let users override these, but make your shared defaults work out-of-the-box.
    os.environ.setdefault("LIVEKIT_URL", DEFAULT_LIVEKIT_URL)
    os.environ.setdefault("LIVEKIT_API_KEY", DEFAULT_LIVEKIT_API_KEY)
    os.environ.setdefault("LIVEKIT_API_SECRET", DEFAULT_LIVEKIT_API_SECRET)
    os.environ.setdefault("LIVEKIT_SIP_TRUNK_ID_OUTBOUND", DEFAULT_SIP_TRUNK_ID)
    # LiveKit Inference flow uses LiveKit auth; keep worker setup zero-config.
    os.environ.setdefault("DEEPGRAM_API_KEY", os.environ["LIVEKIT_API_KEY"])
    # Reduce noisy worker telemetry unless explicitly overridden by user.
    os.environ.setdefault("LIVEKIT_LOG_LEVEL", "WARNING")


def _configure_logging() -> None:
    level_name = os.getenv("LIVEKIT_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.getLogger("livekit").setLevel(level)
    logging.getLogger("livekit.agents").setLevel(level)
    logging.getLogger("livekit.api").setLevel(level)


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


def _history_text(turns: list[dict[str, str]], max_turns: int = 8) -> str:
    clipped = turns[-max_turns:]
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in clipped if t["text"].strip())


def _load_prior_predictor_class():
    try:
        from fstt_priors import PriorPredictor  # type: ignore
        return PriorPredictor
    except ImportError:
        this_dir = Path(__file__).resolve().parent
        if str(this_dir) not in sys.path:
            sys.path.insert(0, str(this_dir))
        from fstt_priors import PriorPredictor  # type: ignore
        return PriorPredictor


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

    def write_vtt(self, json_path: Path) -> Path | None:
        turn_events = [e for e in self.events if e.get("type") == "turn"]
        if not turn_events:
            return None

        def _parse_time(iso_ts: str) -> datetime:
            return datetime.fromisoformat(iso_ts)

        def _fmt_vtt_time(seconds: float) -> str:
            seconds = max(seconds, 0.0)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

        base_dt = _parse_time(turn_events[0]["time"])
        segments: list[tuple[float, float, str]] = []

        for idx, ev in enumerate(turn_events):
            payload = ev.get("payload", {})
            role = str(payload.get("role", "unknown")).upper()
            text = str(payload.get("text", "")).strip()
            if not text:
                continue

            start_dt = _parse_time(ev["time"])
            if idx + 1 < len(turn_events):
                next_dt = _parse_time(turn_events[idx + 1]["time"])
                end_dt = next_dt
            else:
                end_dt = start_dt + timedelta(seconds=4)

            start_s = (start_dt - base_dt).total_seconds()
            end_s = (end_dt - base_dt).total_seconds()
            if end_s <= start_s + 0.2:
                end_s = start_s + 2.0

            segments.append((start_s, end_s, f"{role}: {text}"))

        cues: list[str] = ["WEBVTT", ""]
        for start_s, end_s, text in segments:
            cues.append(f"{_fmt_vtt_time(start_s)} --> {_fmt_vtt_time(end_s)}")
            cues.append(text)
            cues.append("")

        vtt_path = json_path.with_suffix(".vtt")
        vtt_path.write_text("\n".join(cues), encoding="utf-8")
        return vtt_path

    def write_srt(self, json_path: Path) -> Path | None:
        turn_events = [e for e in self.events if e.get("type") == "turn"]
        if not turn_events:
            return None

        def _parse_time(iso_ts: str) -> datetime:
            return datetime.fromisoformat(iso_ts)

        def _fmt_srt_time(seconds: float) -> str:
            seconds = max(seconds, 0.0)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        base_dt = _parse_time(turn_events[0]["time"])
        cues: list[str] = []
        cue_idx = 1

        for idx, ev in enumerate(turn_events):
            payload = ev.get("payload", {})
            role = str(payload.get("role", "unknown")).upper()
            text = str(payload.get("text", "")).strip()
            if not text:
                continue

            start_dt = _parse_time(ev["time"])
            if idx + 1 < len(turn_events):
                next_dt = _parse_time(turn_events[idx + 1]["time"])
                end_dt = next_dt
            else:
                end_dt = start_dt + timedelta(seconds=4)

            start_s = (start_dt - base_dt).total_seconds()
            end_s = (end_dt - base_dt).total_seconds()
            if end_s <= start_s + 0.2:
                end_s = start_s + 2.0

            cues.append(str(cue_idx))
            cues.append(f"{_fmt_srt_time(start_s)} --> {_fmt_srt_time(end_s)}")
            cues.append(f"{role}: {text}")
            cues.append("")
            cue_idx += 1

        srt_path = json_path.with_suffix(".srt")
        srt_path.write_text("\n".join(cues), encoding="utf-8")
        return srt_path

    def write_player_html(self, json_path: Path, audio_path: Path, vtt_path: Path | None) -> Path:
        track_tag = ""
        if vtt_path is not None:
            track_tag = f'<track kind="captions" src="{vtt_path.name}" srclang="en" label="Transcript" default>'
        html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Call Playback</title></head>
<body>
  <h3>{json_path.stem}</h3>
  <audio controls style="width: 100%;">
    <source src="{audio_path.name}" type="audio/ogg">
    {track_tag}
  </audio>
</body>
</html>
"""
        html_path = json_path.with_suffix(".player.html")
        html_path.write_text(html, encoding="utf-8")
        return html_path


def _build_agent_instructions() -> str:
    return (
        "You are taking phone orders for a Chinese restaurant. "
        "Keep replies short and natural for a phone call. "
        "Ask clarifying questions only when needed. "
        "Confirm the order details before ending."
    )


def _build_stt(use_injection: bool):
    stt_model = os.getenv("LK_STT_MODEL", DEFAULT_STT_MODEL)
    direct_dg = os.getenv("AB_USE_DIRECT_DEEPGRAM", "0") == "1"
    if use_injection and direct_dg:
        # Direct Deepgram mode enables dynamic keyterm updates, but requires a real Deepgram key.
        return deepgram.STT(
            model="nova-3",
            language=os.getenv("LK_STT_LANGUAGE", DEFAULT_STT_LANGUAGE),
            punctuate=True,
            smart_format=True,
            interim_results=True,
            keyterm=[],
        )
    # LiveKit Inference mode (no direct Deepgram key required).
    return stt_model


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
    max_terms = int(metadata.get("max_terms", 10))
    topk = int(metadata.get("topk", 30))
    sip_identity = str(metadata.get("sip_participant_identity", "")).strip() or None
    max_call_seconds = int(metadata.get("max_call_seconds", 900))

    recorder = TranscriptRecorder(
        room_name=ctx.room.name,
        call_to=call_to,
        inject_priors=inject_priors,
        output_dir=output_dir,
    )

    predictor = None
    if inject_priors:
        if not model_dir:
            recorder.add_event(
                "warning",
                {"message": "inject_priors=true but model_dir missing; disabling injection"},
            )
            inject_priors = False
        else:
            PriorPredictor = _load_prior_predictor_class()
            predictor = PriorPredictor(str(model_dir), device="cpu")

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
        role = str(getattr(item, "role", "unknown"))
        text = _coerce_text(getattr(item, "content", None))
        recorder.add_turn(role, text)

        if not inject_priors or predictor is None:
            return

        history = _history_text(recorder.history_turns, max_turns=8)
        if not history.strip():
            return
        try:
            terms = predictor.predict(history, max_terms=max_terms, topk=topk)
        except Exception as exc:
            recorder.add_event("prior_prediction_error", {"error": str(exc)})
            return

        if hasattr(stt, "update_options"):
            stt.update_options(keyterm=terms)
        else:
            recorder.add_event(
                "keyterm_update_skipped",
                {
                    "message": "STT is running via LiveKit Inference string model; dynamic keyterm updates are not applied in this mode.",
                    "predicted_terms_count": len(terms),
                },
            )
        recorder.log_terms(terms, history)
        recorder.add_event("deepgram_keyterm_update", {"count": len(terms), "terms": terms})

    await session.start(
        agent=agent,
        room=ctx.room,
        record={"audio": True, "logs": False, "traces": False, "transcript": False},
    )

    # Wait for the PSTN callee to appear, then greet immediately.
    if sip_identity:
        await ctx.wait_for_participant(identity=sip_identity)
    else:
        await ctx.wait_for_participant()

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
        vtt_path = recorder.write_vtt(out_path)
        srt_path = recorder.write_srt(out_path)

        session_audio = Path(ctx.session_directory) / "audio.ogg"
        audio_out_path: Path | None = None
        player_path: Path | None = None
        if session_audio.is_file():
            audio_out_path = out_path.with_suffix(".ogg")
            shutil.copy2(session_audio, audio_out_path)
            player_path = recorder.write_player_html(out_path, audio_out_path, vtt_path)

        print(f"[ab-test] transcript saved: {out_path}")
        if audio_out_path:
            print(f"[ab-test] audio saved: {audio_out_path}")
        if vtt_path:
            print(f"[ab-test] captions saved: {vtt_path}")
        if srt_path:
            print(f"[ab-test] srt saved: {srt_path}")
        if player_path:
            print(f"[ab-test] player saved: {player_path}")


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
        "output_dir": args.output_dir,
        "greeting": args.greeting,
        "sip_participant_identity": sip_participant_identity,
        "max_call_seconds": args.max_call_seconds,
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

    sub.add_parser("worker", help="Run the LiveKit agent worker")

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
        help="Enable retrieval-based keyterm injection",
    )
    dial.add_argument(
        "--model-dir",
        default=os.getenv("FSTT_MODEL_DIR", ""),
        help="Model directory containing best_model/ and shared_index/",
    )
    dial.add_argument("--max-terms", type=int, default=10, help="Max Deepgram keyterms")
    dial.add_argument("--topk", type=int, default=30, help="Top-k retrieval depth")
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
    return parser


def main() -> None:
    _apply_default_env()
    _configure_logging()
    args = _parser().parse_args()

    if args.mode == "worker":
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
