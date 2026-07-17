"""Pipecat voice bot with openWakeWord gating — clean rebuild.

Flow (matches the state diagram):

    LOCKED   openWakeWord watches the mic locally. No audio reaches
             Speechmatics (zero STT cost while idle).
       │ wake word detected
       ▼
    OPEN     All mic audio streams to Speechmatics. ADAPTIVE turn detection
             (with a generous silence trigger) decides when you finished a
             sentence, so breathing pauses don't cut you off. If you never
             start speaking, a relock timer sends us back to LOCKED.
       │ sentence completion detected (UserStoppedSpeaking)
       ▼
    MUTED    The LLM thinks and the bot speaks. Mic audio is replaced with
             SILENCE toward Speechmatics (keeps its audio timeline continuous
             so half-heard words flush out now and get dropped, instead of
             leaking into the next turn). openWakeWord still watches the real
             mic: saying the wake word interrupts the bot (barge-in).
       │ TTS finished (BotStoppedSpeaking)
       ▼
    OPEN     Mic reopens for FOLLOWUP_SECONDS so you can reply without the
             wake word. No input -> LOCKED.

Frame-hygiene rules learned the hard way (see SttGateBridge):

1. Interim transcriptions are NEVER forwarded. The aggregator's turn-stop
   strategy holds back LLM inference while an interim is pending until its
   final arrives — and if that final gets dropped as stale, the turn
   deadlocks. Finals alone carry the text, so interims are safe to drop.
2. A turn's final transcript can arrive AFTER its UserStoppedSpeaking frame
   (the STT pushes finals through an internal queue). A 0.5s grace window
   admits exactly ONE late final — only if the turn has no text yet.
   Anything more would fire a duplicate LLM inference for the same turn.
3. Any other transcript arriving while the gate isn't OPEN is stale buffered
   audio; forwarding it would become a phantom LLM turn. Drop it.
4. Turn signals (UserStarted/StoppedSpeaking) caused by that stale audio are
   dropped too, so they can't restart timers or reopen the grace window.
5. If the LLM/TTS never responds, a watchdog relocks the gate instead of
   leaving the mic muted forever.

.env: OPENAI_API_KEY, SPEECHMATICS_API_KEY, LITY_BASE_URL, LITY_API_KEY,
LITY_MODEL, and optionally WAKE_WORD, WAKE_THRESHOLD, WAKE_LISTEN_SECONDS,
FOLLOWUP_SECONDS, EOU_SILENCE_TRIGGER, PROCESSING_TIMEOUT_SECONDS,
AUDIO_INPUT_DEVICE_INDEX, AUDIO_OUTPUT_DEVICE_INDEX.
"""

import asyncio
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("openwakeword").setLevel(logging.ERROR)

import numpy as np
import soxr
from dotenv import load_dotenv
from loguru import logger
from openwakeword.model import Model as WakeWordModel

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
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
    """LOCKED / OPEN / MUTED state machine sitting between mic and STT.

    Owns all audio-level decisions: wake word detection, forwarding audio,
    silence injection, barge-in, relock and watchdog timers. Turn signals
    arrive via :class:`SttGateBridge` (which sits downstream of the STT).
    """

    def __init__(self):
        super().__init__()
        self._wakeword = os.getenv("WAKE_WORD", "hey_jarvis")
        self._threshold = float(os.getenv("WAKE_THRESHOLD", "0.5"))
        self._wake_listen_seconds = float(os.getenv("WAKE_LISTEN_SECONDS", "8"))
        self._followup_seconds = float(os.getenv("FOLLOWUP_SECONDS", "3"))
        self._mute_timeout = float(os.getenv("PROCESSING_TIMEOUT_SECONDS", "30"))

        _ensure_wakeword_models(self._wakeword)
        self._oww = WakeWordModel(
            wakeword_models=[self._wakeword], inference_framework="onnx"
        )
        self._pending = np.zeros(0, dtype=np.int16)  # buffer for 80 ms chunking

        self._state = "LOCKED"
        self._user_speaking = False
        self._relock_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None

    @property
    def is_open(self) -> bool:
        """True when we're accepting user speech (mic → Speechmatics)."""
        return self._state == "OPEN"

    # ---- pipeline entry point ------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_audio(frame)
            return

        # Bot speaking frames also travel upstream, so we see them here.
        if isinstance(frame, BotStartedSpeakingFrame):
            self._cancel_watchdog()  # the bot responded; the turn wasn't lost
            self._mute("bot speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Only on a natural end of bot speech. After a barge-in we're
            # already OPEN with the full wake window; the interrupted bot's
            # trailing BotStopped must not shrink it.
            if self._state == "MUTED":
                self._open(self._followup_seconds, "bot finished")

        await self.push_frame(frame, direction)

    # ---- audio routing (the heart of the gate) -------------------------------
    async def _handle_audio(self, frame: InputAudioRawFrame):
        if self._state == "OPEN":
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)  # -> STT

        elif self._state == "LOCKED":
            if self._detect_wakeword(frame):
                self._open(self._wake_listen_seconds, "wake word")

        elif self._state == "MUTED":
            # Barge-in: the wake word cuts the bot off and reopens the mic.
            if self._detect_wakeword(frame):
                logger.info("🙋 Wake word during bot response — interrupting")
                await self.broadcast_interruption()
                self._open(self._wake_listen_seconds, "barge-in")
                return
            # Otherwise feed SILENCE to Speechmatics: the mic stays deaf, but
            # the audio timeline stays continuous, so words that were in
            # flight when we muted finalize now (and the bridge drops them)
            # instead of leaking out as a phantom turn when the mic reopens.
            silence = InputAudioRawFrame(
                audio=b"\x00" * len(frame.audio),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            await self.push_frame(silence, FrameDirection.DOWNSTREAM)

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

    # ---- state transitions ---------------------------------------------------
    def _open(self, relock_after: float, reason: str):
        self._state = "OPEN"
        self._user_speaking = False
        self._cancel_relock()
        self._cancel_watchdog()
        self._relock_task = asyncio.create_task(self._relock_timer(relock_after))
        logger.info(f"🟢 Open ({reason}) — mic → Speechmatics")

    def _mute(self, reason: str):
        if self._state == "MUTED":
            return
        self._state = "MUTED"
        self._cancel_relock()
        logger.info(f"🔇 Muted ({reason}) — bot's turn")

    def _lock(self, reason: str):
        self._state = "LOCKED"
        self._cancel_relock()
        self._cancel_watchdog()
        self._pending = np.zeros(0, dtype=np.int16)
        self._oww.reset()
        spoken = self._wakeword.replace("_", " ")
        logger.info(f"🔒 Locked ({reason}) — say “{spoken}” to wake")

    # ---- called by SttGateBridge ---------------------------------------------
    def notify_user_started(self):
        """User began speaking — keep the mic open, cancel the relock timer."""
        self._user_speaking = True
        self._cancel_relock()

    def notify_user_stopped(self):
        """User finished a sentence — bot's turn now."""
        self._user_speaking = False
        self._mute("turn ended")
        # If no bot speech follows (turn lost, backend down), don't stay
        # muted forever — relock so the wake word works again.
        self._start_watchdog()

    # ---- timers --------------------------------------------------------------
    async def _relock_timer(self, timeout: float):
        try:
            await asyncio.sleep(timeout)
            if not self._user_speaking:
                self._lock("no speech")
        except asyncio.CancelledError:
            pass

    def _cancel_relock(self):
        if self._relock_task and not self._relock_task.done():
            self._relock_task.cancel()
        self._relock_task = None

    def _start_watchdog(self):
        self._cancel_watchdog()
        self._watchdog_task = asyncio.create_task(self._watchdog_timer())

    async def _watchdog_timer(self):
        try:
            await asyncio.sleep(self._mute_timeout)
            if self._state == "MUTED":
                logger.warning(
                    f"⚠️  No bot response after {self._mute_timeout:.0f}s — relocking"
                )
                self._lock("no response")
        except asyncio.CancelledError:
            pass

    def _cancel_watchdog(self):
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
        self._watchdog_task = None


class SttGateBridge(FrameProcessor):
    """Sits right after the STT: filters its output and drives the gate.

    Implements the frame-hygiene rules from the module docstring — exactly
    one LLM inference per spoken turn, no phantom turns from stale audio.
    """

    # The aggregator triggers inference 0.5s after the last final transcript;
    # a late final admitted within this window merges into the current turn.
    # Keep this <= that strategy timeout (0.5s in pipecat 1.5.0).
    GRACE_SECONDS = 0.5

    def __init__(self, gate: WakeWordGate):
        super().__init__()
        self._gate = gate
        self._turn_stopped_at: float | None = None
        self._turn_has_text = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            return  # rule 1: never forward interims

        if isinstance(frame, TranscriptionFrame):
            if self._gate.is_open:
                self._turn_has_text = True
            else:
                in_grace = (
                    not self._turn_has_text  # rule 2: only if turn is empty
                    and self._turn_stopped_at is not None
                    and time.monotonic() - self._turn_stopped_at <= self.GRACE_SECONDS
                )
                if not in_grace:
                    logger.info(f"🗑️ Dropped stale transcript: {frame.text!r}")
                    return  # rule 3: stale audio, not a new turn
                self._turn_has_text = True  # rule 2: admit exactly one

        elif isinstance(frame, UserStartedSpeakingFrame):
            if not self._gate.is_open:
                return  # rule 4: turn-start from stale audio
            self._turn_has_text = False  # a new turn begins, no text yet
            self._gate.notify_user_started()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if not self._gate.is_open:
                return  # rule 4: turn-end from stale audio
            self._turn_stopped_at = time.monotonic()
            self._gate.notify_user_stopped()

        await self.push_frame(frame, direction)


class TranscriptLogObserver(BaseObserver):
    """Prints 'You:' / 'Bot:' lines plus per-turn LLM/error diagnostics."""

    def __init__(self):
        super().__init__()
        self._reply = ""

    async def on_push_frame(self, data: FramePushed):
        src, frame = data.source, data.frame
        if isinstance(frame, TranscriptionFrame) and isinstance(src, STTService):
            logger.info(f"🧑 You: {frame.text}")
        elif isinstance(frame, LLMContextFrame) and "User" in type(src).__name__:
            # Exactly one per spoken turn is healthy; two = duplicate
            # trigger; zero = the turn was lost before the LLM.
            logger.info("🧠 LLM inference triggered")
        elif isinstance(frame, ErrorFrame):
            logger.error(f"❌ Pipeline error from {src}: {frame.error}")
        elif isinstance(frame, LLMTextFrame) and isinstance(src, LLMService):
            self._reply += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame) and isinstance(src, LLMService):
            if self._reply.strip():
                logger.info(f"🤖 Bot: {self._reply.strip()}")
            self._reply = ""


def _env_int(name: str) -> int | None:
    val = os.getenv(name, "").strip()
    return int(val) if val else None


async def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    speechmatics_key = os.getenv("SPEECHMATICS_API_KEY")
    if not openai_key or not speechmatics_key:
        raise RuntimeError(
            "Missing API keys. Set OPENAI_API_KEY and SPEECHMATICS_API_KEY in .env"
        )

    lity_base_url = os.getenv("LITY_BASE_URL", "http://localhost:8321/v1").rstrip("/")
    lity_api_key = os.getenv("LITY_API_KEY", "")
    lity_model = os.getenv("LITY_MODEL", "lity")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=_env_int("AUDIO_INPUT_DEVICE_INDEX"),
            output_device_index=_env_int("AUDIO_OUTPUT_DEVICE_INDEX"),
        )
    )

    gate = WakeWordGate()
    bridge = SttGateBridge(gate)

    stt = SpeechmaticsSTTService(
        api_key=speechmatics_key,
        settings=SpeechmaticsSTTService.Settings(
            language=Language.EN,
            # The default (EXTERNAL) waits for a separate VAD to end turns,
            # which this pipeline doesn't have — use the server's built-in
            # endpointing instead.
            turn_detection_mode=SpeechmaticsSTTService.TurnDetectionMode.ADAPTIVE,
            # Max pause that still counts as "thinking" rather than "done".
            # ADAPTIVE ends turns sooner when the sentence sounds complete,
            # so a high value mostly costs latency when you trail off.
            end_of_utterance_silence_trigger=float(
                os.getenv("EOU_SILENCE_TRIGGER", "1.5")
            ),
        ),
    )

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

    # Lity keeps conversation memory server-side; each turn just carries the
    # latest user utterance, so the context starts empty.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            gate,                     # wake word / mute / barge-in
            stt,
            bridge,                   # frame hygiene + turn signals to gate
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[TranscriptLogObserver()],
        # We can sit LOCKED (no BotSpeaking/UserSpeaking frames) for long
        # stretches waiting for the wake word — that's normal, not a stuck
        # pipeline, so disable the idle-cancel watchdog entirely.
        idle_timeout_secs=None,
    )

    runner = PipelineRunner(handle_sigint=True)

    spoken = os.getenv("WAKE_WORD", "hey_jarvis").replace("_", " ")
    logger.info(f"Bot is running. Say “{spoken}” to wake it. Press Ctrl+C to stop.")
    logger.info(f"🔒 Locked — say “{spoken}” to wake")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
