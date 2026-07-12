"""Pipecat voice bot with an openWakeWord gate.

Pipeline:
    mic -> [WakeWordGate] -> Speechmatics STT -> [SttGateBridge]
        -> LLM -> TTS -> speakers

Wake-word gating (see the state diagram this implements):

    LOCKED      openWakeWord listens locally. Mic audio is NOT forwarded to
                Speechmatics, so there is zero STT cost while idle. Say the wake
                word ("hey jarvis") to wake.
       │  wake word detected
       ▼
    LISTENING   Mic audio is streamed to Speechmatics. A relock timer runs; if
                you don't start speaking before it fires, we go back to LOCKED.
       │  Speechmatics reports you finished a sentence (UserStoppedSpeaking)
       ▼
    PROCESSING  Mic is muted while the LLM thinks and the bot speaks. This is
                also what stops the bot from hearing its own voice.
       │  bot finished speaking (BotStoppedSpeaking)
       ▼
    LISTENING   Re-open the mic for `FOLLOWUP_SECONDS` (default 3s) so you can
                reply without repeating the wake word. Silence -> LOCKED.

Why a custom audio gate instead of pipecat's built-in WakeCheckFilter: that
filter matches a wake *phrase* in the transcription, which means the audio was
already sent to (and billed by) the STT. Gating on raw audio here means nothing
reaches Speechmatics until the wake word fires locally.
"""

import asyncio
import logging
import os
import sys
import warnings

# Pipecat 1.5.0 soft-deprecates PipelineTask/PipelineRunner in favor of the
# newer Worker API, but the task/runner pattern below is still the documented
# mainstream one and works through 2.0. Silence the noise for a clean console.
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("openwakeword").setLevel(logging.ERROR)

import httpx
import numpy as np
import soxr
from dotenv import load_dotenv
from loguru import logger
from openwakeword.model import Model as WakeWordModel

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

load_dotenv()

# Keep the console readable: pipecat is very chatty at DEBUG.
logger.remove()
logger.add(sys.stderr, level="INFO")

# openWakeWord expects 16 kHz mono int16 audio, processed in 80 ms chunks.
OWW_SAMPLE_RATE = 16000
OWW_CHUNK = 1280  # samples (80 ms @ 16 kHz)


def _ensure_wakeword_models(wakeword: str):
    """Download openWakeWord's base + wakeword models once, if not present."""
    import openwakeword
    from openwakeword.utils import download_models

    models_dir = os.path.join(
        os.path.dirname(openwakeword.__file__), "resources", "models"
    )
    needed = ["melspectrogram.onnx", "embedding_model.onnx", f"{wakeword}_v0.1.onnx"]
    if not all(os.path.exists(os.path.join(models_dir, n)) for n in needed):
        logger.info("Downloading openWakeWord models (one-time)…")
        download_models()


class WakeWordGate(FrameProcessor):
    """Gates mic audio: nothing reaches the STT until the wake word is heard.

    Sits directly after ``transport.input()``. Drives a LOCKED / LISTENING /
    PROCESSING state machine (see module docstring). User-turn signals arrive
    from :class:`SttGateBridge`, which is downstream of the STT.
    """

    def __init__(
        self,
        *,
        wakeword: str,
        threshold: float,
        wake_listen_seconds: float,
        followup_seconds: float,
    ):
        super().__init__()
        self._wakeword = wakeword
        self._threshold = threshold
        self._wake_listen_seconds = wake_listen_seconds
        self._followup_seconds = followup_seconds

        _ensure_wakeword_models(wakeword)
        self._oww = WakeWordModel(
            wakeword_models=[wakeword], inference_framework="onnx"
        )
        self._pending = np.zeros(0, dtype=np.int16)  # buffer for 80 ms chunking

        self._state = "LOCKED"
        self._user_speaking = False
        self._relock_task: asyncio.Task | None = None

    @property
    def is_idle(self) -> bool:
        """True only when locked (wake-word idle) — safe to poll for pushes."""
        return self._state == "LOCKED"

    # ---- pipeline entry point -------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Mic audio (flows downstream). We decide whether to forward it.
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_audio(frame)
            return

        # Bot speaking frames travel upstream too, so we can see them here and
        # use them to drive the state machine.
        if isinstance(frame, BotStartedSpeakingFrame):
            self._enter_processing()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._enter_listening(self._followup_seconds, "bot finished")

        await self.push_frame(frame, direction)

    # ---- audio handling -------------------------------------------------------
    async def _handle_audio(self, frame: InputAudioRawFrame):
        if self._state == "LISTENING":
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)  # -> STT
        elif self._state == "LOCKED":
            if self._detect_wakeword(frame):
                await self._enter_listening(self._wake_listen_seconds, "wake word")
        # PROCESSING: drop the frame (mic muted while bot responds).

    def _detect_wakeword(self, frame: InputAudioRawFrame) -> bool:
        samples = self._to_16k_mono(frame)
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= OWW_CHUNK:
            chunk = self._pending[:OWW_CHUNK]
            self._pending = self._pending[OWW_CHUNK:]
            scores = self._oww.predict(chunk)
            if scores.get(self._wakeword, 0.0) >= self._threshold:
                self._pending = np.zeros(0, dtype=np.int16)
                self._oww.reset()
                return True
        return False

    @staticmethod
    def _to_16k_mono(frame: InputAudioRawFrame) -> np.ndarray:
        samples = np.frombuffer(frame.audio, dtype=np.int16)
        if frame.num_channels > 1:  # downmix to mono
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        if frame.sample_rate != OWW_SAMPLE_RATE:  # resample to 16 kHz
            resampled = soxr.resample(
                samples.astype(np.float32), frame.sample_rate, OWW_SAMPLE_RATE
            )
            samples = np.clip(resampled, -32768, 32767).astype(np.int16)
        return samples

    # ---- state transitions ----------------------------------------------------
    async def _enter_listening(self, timeout: float, reason: str):
        self._state = "LISTENING"
        self._user_speaking = False
        self._cancel_relock()
        logger.info(f"🟢 Listening ({reason}) — mic → Speechmatics")
        self._relock_task = asyncio.create_task(self._relock_after(timeout))

    def _enter_processing(self):
        if self._state == "PROCESSING":
            return
        self._state = "PROCESSING"
        self._cancel_relock()
        logger.info("⏳ Processing — mic muted")

    async def _lock(self, reason: str):
        self._state = "LOCKED"
        self._cancel_relock()
        self._pending = np.zeros(0, dtype=np.int16)
        self._oww.reset()
        logger.info(f"🔒 Locked ({reason}) — say “{self._wakeword.replace('_', ' ')}” to wake")

    async def _relock_after(self, timeout: float):
        try:
            await asyncio.sleep(timeout)
            if not self._user_speaking:
                await self._lock("no speech")
        except asyncio.CancelledError:
            pass

    def _cancel_relock(self):
        if self._relock_task and not self._relock_task.done():
            self._relock_task.cancel()
        self._relock_task = None

    # ---- called by SttGateBridge (downstream of the STT) ----------------------
    def notify_user_started(self):
        """User began speaking — keep the mic open, cancel any relock."""
        self._user_speaking = True
        self._cancel_relock()

    def notify_user_stopped(self):
        """User finished a sentence — mute the mic while the bot responds."""
        self._user_speaking = False
        self._enter_processing()


class SttGateBridge(FrameProcessor):
    """Relays the STT's user-turn frames back to the WakeWordGate.

    UserStarted/Stopped frames only flow downstream, so the gate (which is
    upstream of the STT) can't see them without this bridge.
    """

    def __init__(self, gate: WakeWordGate):
        super().__init__()
        self._gate = gate

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._gate.notify_user_started()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._gate.notify_user_stopped()
        await self.push_frame(frame, direction)


class ConversationLogObserver(BaseObserver):
    """Prints a clean 'You:' / 'Bot:' transcript of the conversation."""

    def __init__(self):
        super().__init__()
        self._reply = ""

    async def on_push_frame(self, data: FramePushed):
        src, frame = data.source, data.frame
        if isinstance(frame, TranscriptionFrame) and isinstance(src, STTService):
            logger.info(f"🧑 You: {frame.text}")
        elif isinstance(frame, LLMTextFrame) and isinstance(src, LLMService):
            self._reply += frame.text  # LLM streams tokens; buffer them
        elif isinstance(frame, LLMFullResponseEndFrame) and isinstance(src, LLMService):
            if self._reply.strip():
                logger.info(f"🤖 Bot: {self._reply.strip()}")
            self._reply = ""


async def lity_health_check(base_url: str, headers: dict):
    """Ping GET /v1/models so a bad URL/host is obvious at startup (non-fatal)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{base_url}/models", headers=headers)
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
        logger.info(f"✅ Lity reachable at {base_url} (models: {ids})")
    except Exception as e:
        logger.warning(
            f"⚠️  Lity not reachable at {base_url} ({e}). "
            "Wake word / STT / TTS still work, but replies will fail until it's up."
        )


async def poll_pending(
    task: PipelineTask,
    gate: "WakeWordGate",
    base_url: str,
    headers: dict,
    interval: float,
):
    """While idle (wake-word locked), poll GET /v1/voice/pending and speak pushes.

    Only polls when the gate is idle: mid-conversation, Lity piggybacks pending
    messages onto the next chat/completions reply, so polling then would consume
    them out from under that flow. Each spoken message drives the bot through the
    normal PROCESSING -> follow-up-LISTENING states, so the user can respond
    (e.g. approve) without repeating the wake word.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(interval)
            if not gate.is_idle:
                continue  # mid-conversation: pending rides in on the next turn
            try:
                r = await client.get(f"{base_url}/voice/pending", headers=headers)
                r.raise_for_status()
                messages = r.json().get("messages", [])
            except Exception as e:
                logger.warning(f"pending poll failed: {e}")
                continue
            if messages:
                logger.info(f"📨 {len(messages)} pending message(s) from Lity")
                # append_to_context=False: these are standalone announcements;
                # Lity owns conversation memory, not our local context.
                await task.queue_frames(
                    [TTSSpeakFrame(m, append_to_context=False) for m in messages]
                )


async def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    speechmatics_key = os.getenv("SPEECHMATICS_API_KEY")
    if not openai_key or not speechmatics_key:
        raise RuntimeError(
            "Missing API keys. Set OPENAI_API_KEY and SPEECHMATICS_API_KEY in .env"
        )

    wakeword = os.getenv("WAKE_WORD", "hey_jarvis")
    wake_threshold = float(os.getenv("WAKE_THRESHOLD", "0.5"))
    wake_listen_seconds = float(os.getenv("WAKE_LISTEN_SECONDS", "8"))
    followup_seconds = float(os.getenv("FOLLOWUP_SECONDS", "3"))

    # Lity: our custom OpenAI-compatible LLM backend (turns + proactive pushes).
    lity_base_url = os.getenv("LITY_BASE_URL", "http://localhost:8321/v1").rstrip("/")
    lity_api_key = os.getenv("LITY_API_KEY", "")
    lity_model = os.getenv("LITY_MODEL", "lity")
    lity_poll_seconds = float(os.getenv("LITY_POLL_SECONDS", "4"))
    lity_headers = {"Content-Type": "application/json"}
    if lity_api_key:
        lity_headers["Authorization"] = f"Bearer {lity_api_key}"

    # --- Transport: local mic + speakers (no VAD; Speechmatics endpoints) ------
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    # --- Wake-word gate + STT bridge ------------------------------------------
    gate = WakeWordGate(
        wakeword=wakeword,
        threshold=wake_threshold,
        wake_listen_seconds=wake_listen_seconds,
        followup_seconds=followup_seconds,
    )
    stt_bridge = SttGateBridge(gate)

    # --- Speech-to-text: Speechmatics (server-side turn detection) -------------
    stt = SpeechmaticsSTTService(
        api_key=speechmatics_key,
        settings=SpeechmaticsSTTService.Settings(
            language=Language.EN,
            turn_detection_mode=SpeechmaticsSTTService.TurnDetectionMode.ADAPTIVE,
        ),
    )

    # --- LLM: Lity (OpenAI-compatible). Reads only the last user message and
    #     owns conversation memory server-side, so we send no system prompt or
    #     history. TTS stays OpenAI; Lity returns already-speakable text. --------
    llm = OpenAILLMService(
        api_key=lity_api_key or "lity",  # OpenAI client needs a non-empty key
        base_url=lity_base_url,
        settings=OpenAILLMService.Settings(model=lity_model),
    )
    tts = OpenAITTSService(
        api_key=openai_key,
        settings=OpenAITTSService.Settings(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.getenv("OPENAI_VOICE_ID", "alloy"),
        ),
    )

    # Empty context: Lity ignores history/system prompt and keeps its own memory,
    # so each turn just carries the latest user utterance.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),            # mic audio in
            gate,                         # wake-word gate (may drop audio)
            stt,                          # audio -> text (only when unlocked)
            stt_bridge,                   # relay user-turn events to the gate
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),           # audio out to speakers
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[ConversationLogObserver()],
    )

    runner = PipelineRunner(handle_sigint=True)

    await lity_health_check(lity_base_url, lity_headers)

    spoken = wakeword.replace("_", " ")
    logger.info(
        f"Bot is running. Say “{spoken}” to wake it. Press Ctrl+C to stop."
    )
    logger.info(f"🔒 Locked — say “{spoken}” to wake")

    # Run the pipeline and the idle poller together; stop the poller on exit.
    poller = asyncio.create_task(
        poll_pending(task, gate, lity_base_url, lity_headers, lity_poll_seconds)
    )
    try:
        await runner.run(task)
    finally:
        poller.cancel()


if __name__ == "__main__":
    asyncio.run(main())
