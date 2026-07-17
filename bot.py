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
                End-of-turn is Speechmatics' ADAPTIVE detection with a generous
                silence trigger (EOU_SILENCE_TRIGGER, default 1.5s), so pausing
                to breathe mid-thought doesn't end your turn.
       │  Speechmatics reports you finished your turn (UserStoppedSpeaking)
       ▼
    PROCESSING  Mic audio is replaced with silence while the LLM thinks and the
                bot speaks (stops self-hearing, keeps the STT stream continuous
                so leftover words flush out and get dropped). openWakeWord still
                runs on the real mic: saying the wake word here interrupts the
                bot mid-sentence (barge-in) and reopens the mic.
       │  bot finished speaking (BotStoppedSpeaking)
       ▼
    LISTENING   Re-open the mic for `FOLLOWUP_SECONDS` (default 5s) so you can
                reply without repeating the wake word. Silence -> LOCKED.

Why a custom audio gate instead of pipecat's built-in WakeCheckFilter: that
filter matches a wake *phrase* in the transcription, which means the audio was
already sent to (and billed by) the STT. Gating on raw audio here means nothing
reaches Speechmatics until the wake word fires locally.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
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
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
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
        # Watchdog: if we enter PROCESSING but the bot never starts speaking
        # (turn lost, LLM/backend down), relock instead of staying muted forever.
        self._processing_watchdog: asyncio.Task | None = None
        self._processing_timeout = float(os.getenv("PROCESSING_TIMEOUT_SECONDS", "30"))

    @property
    def is_idle(self) -> bool:
        """True only when locked (wake-word idle) — safe to poll for pushes."""
        return self._state == "LOCKED"

    @property
    def is_listening(self) -> bool:
        """True when we're accepting user speech (mic → STT)."""
        return self._state == "LISTENING"

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
            # The bot is responding — the turn wasn't lost; stand down.
            self._cancel_processing_watchdog()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Only transition on a natural end of speech; after a barge-in the
            # gate is already LISTENING with the full wake window and the
            # interrupted bot's trailing BotStopped must not shrink it.
            if self._state == "PROCESSING":
                await self._enter_listening(self._followup_seconds, "bot finished")

        await self.push_frame(frame, direction)

    # ---- audio handling -------------------------------------------------------
    async def _handle_audio(self, frame: InputAudioRawFrame):
        if self._state == "LISTENING":
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)  # -> STT
        elif self._state == "LOCKED":
            if self._detect_wakeword(frame):
                await self._enter_listening(self._wake_listen_seconds, "wake word")
        elif self._state == "PROCESSING":
            # Barge-in: openWakeWord keeps watching the real mic while the bot
            # responds. Saying the wake word cuts the bot off and reopens the
            # mic, so you can talk over it naturally.
            if self._detect_wakeword(frame):
                logger.info("🙋 Wake word during bot response — interrupting")
                await self.broadcast_interruption()
                await self._enter_listening(self._wake_listen_seconds, "barge-in")
                return
            # Otherwise the mic stays muted, but keep the Speechmatics audio
            # timeline continuous by sending silence. Words that were in
            # flight at mute time then finalize *now* (and are dropped as
            # stale by the bridge) instead of sitting in the server's buffer
            # and popping out as a phantom turn the moment the mic reopens.
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

    # ---- state transitions ----------------------------------------------------
    async def _enter_listening(self, timeout: float, reason: str):
        self._state = "LISTENING"
        self._user_speaking = False
        self._cancel_relock()
        self._cancel_processing_watchdog()
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
        self._cancel_processing_watchdog()
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

    def _start_processing_watchdog(self):
        self._cancel_processing_watchdog()
        self._processing_watchdog = asyncio.create_task(self._processing_watchdog_expired())

    async def _processing_watchdog_expired(self):
        try:
            await asyncio.sleep(self._processing_timeout)
            if self._state == "PROCESSING":
                logger.warning(
                    f"⚠️  No bot response after {self._processing_timeout:.0f}s — relocking"
                )
                await self._lock("no response")
        except asyncio.CancelledError:
            pass

    def _cancel_processing_watchdog(self):
        if self._processing_watchdog and not self._processing_watchdog.done():
            self._processing_watchdog.cancel()
        self._processing_watchdog = None

    # ---- called by SttGateBridge (downstream of the STT) ----------------------
    def notify_user_started(self):
        """User began speaking — keep the mic open, cancel any relock."""
        self._user_speaking = True
        self._cancel_relock()

    def notify_user_stopped(self):
        """User finished a sentence — mute the mic while the bot responds."""
        self._user_speaking = False
        self._enter_processing()
        self._start_processing_watchdog()


class SttGateBridge(FrameProcessor):
    """Relays the STT's user-turn frames back to the WakeWordGate.

    UserStarted/Stopped frames only flow downstream, so the gate (which is
    upstream of the STT) can't see them without this bridge.

    It also solves the "stuck words" problem: when the gate mutes the mic
    mid-sentence (Speechmatics ended the turn at a pause), audio already sent
    to Speechmatics sits unfinalized in its buffer. When the mic reopens
    minutes later, new audio flushes that stale segment out as a fresh
    TranscriptionFrame, which would get sent to the LLM as a phantom turn.
    So on every end-of-turn we tell the Speechmatics client to finalize its
    buffer immediately, and we drop any transcription frames that arrive while
    the gate isn't listening, so the flushed leftovers never reach the context
    aggregator.
    """

    # The aggregator's turn-stop strategy waits 0.5s after the last final
    # transcription before triggering the LLM. Finals passed within this grace
    # window after turn-stop always land before that trigger and get merged
    # into the turn; anything later would start a phantom turn, so it's dropped.
    # Keep this <= the strategy timeout (0.5s in pipecat 1.5.0).
    TRANSCRIPT_GRACE_SECONDS = 0.5

    def __init__(self, gate: WakeWordGate, stt: SpeechmaticsSTTService):
        super().__init__()
        self._gate = gate
        self._stt = stt
        self._turn_stopped_at: float | None = None
        self._saw_final_in_turn = False

    def _flush_stt_buffer(self):
        """Force Speechmatics to emit (and let us discard) any buffered audio."""
        client = getattr(self._stt, "_client", None)
        if client is None:
            return
        try:
            client.finalize()
        except Exception as e:
            logger.debug(f"STT buffer flush failed (non-fatal): {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            # Never forward interims. The aggregator's turn-stop strategy
            # blocks LLM inference while an interim is "pending" until the
            # matching final arrives — but finals for audio spoken after the
            # turn ended get dropped below (they're the phantom turns), which
            # would leave the strategy waiting forever. Since only finals
            # carry text into the context, interims are safe to drop always.
            return
        elif isinstance(frame, TranscriptionFrame):
            if self._gate.is_listening:
                self._saw_final_in_turn = True
            else:
                # The STT pushes finals through an internal queue, so a turn's
                # text can arrive here *after* the UserStoppedSpeaking that
                # ended it. Only in that case (no final passed yet — the turn
                # would otherwise be empty and stall) let it through for a
                # short grace period. If the turn already has its text, any
                # late final would fire a SECOND LLM inference for the same
                # turn, so it's dropped.
                in_grace = (
                    not self._saw_final_in_turn
                    and self._turn_stopped_at is not None
                    and time.monotonic() - self._turn_stopped_at <= self.TRANSCRIPT_GRACE_SECONDS
                )
                if not in_grace:
                    logger.info(f"🗑️ Dropping stale transcription: {frame.text!r}")
                    return
                # One late final is the turn's text; close the window so no
                # second final can trigger a duplicate LLM inference.
                self._saw_final_in_turn = True
        elif isinstance(frame, UserStartedSpeakingFrame):
            if not self._gate.is_listening:
                return  # phantom turn-start from buffered audio
            self._saw_final_in_turn = False  # new turn, no text yet
            self._gate.notify_user_started()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if not self._gate.is_listening:
                # End-of-turn for stale buffered audio (gate already muted or
                # locked). Letting it through would reopen the transcript
                # grace window and restart the processing watchdog.
                return
            self._turn_stopped_at = time.monotonic()
            self._gate.notify_user_stopped()
            # The gate is muted now: force Speechmatics to finalize any audio
            # still in its buffer so it can't leak into the next turn. The
            # resulting stale frames arrive while we're not listening and are
            # dropped above.
            self._flush_stt_buffer()

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
        elif isinstance(frame, LLMContextFrame) and not isinstance(src, LLMService):
            # The user aggregator pushing context = one LLM inference firing.
            # Exactly one of these per spoken turn is healthy; two means a
            # duplicate trigger; zero means the turn was lost upstream.
            logger.info("🧠 LLM inference triggered")
        elif isinstance(frame, ErrorFrame):
            logger.error(f"❌ Pipeline error from {src}: {frame.error}")
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


def _env_int(name: str) -> int | None:
    val = os.getenv(name, "").strip()
    return int(val) if val else None


def list_audio_devices():
    """Print PyAudio device indices/names, e.g. to configure a Pi's mic/speaker."""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        print(f"{'idx':>4}  {'in':>3} {'out':>3}  name")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(
                f"{i:>4}  {info['maxInputChannels']:>3} {info['maxOutputChannels']:>3}  "
                f"{info['name']}"
            )
        try:
            print(f"\nDefault input:  {pa.get_default_input_device_info()['name']}")
        except OSError:
            print("\nDefault input:  (none)")
        try:
            print(f"Default output: {pa.get_default_output_device_info()['name']}")
        except OSError:
            print("Default output: (none)")
        print(
            "\nSet AUDIO_INPUT_DEVICE_INDEX / AUDIO_OUTPUT_DEVICE_INDEX in .env "
            "to override the default."
        )
    finally:
        pa.terminate()


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
    followup_seconds = float(os.getenv("FOLLOWUP_SECONDS", "5"))

    # Lity: our custom OpenAI-compatible LLM backend (turns + proactive pushes).
    lity_base_url = os.getenv("LITY_BASE_URL", "http://localhost:8321/v1").rstrip("/")
    lity_api_key = os.getenv("LITY_API_KEY", "")
    lity_model = os.getenv("LITY_MODEL", "lity")
    lity_poll_seconds = float(os.getenv("LITY_POLL_SECONDS", "4"))
    lity_headers = {"Content-Type": "application/json"}
    if lity_api_key:
        lity_headers["Authorization"] = f"Bearer {lity_api_key}"

    # --- Transport: local mic + speakers (no VAD; Speechmatics endpoints) ------
    # On boards with multiple audio devices (e.g. a Pi with HDMI + headphone
    # jack + USB mic), the ALSA default is not always the one you want. Run
    # `bot.py --list-devices` to see indices, then set these in .env.
    input_device_index = _env_int("AUDIO_INPUT_DEVICE_INDEX")
    output_device_index = _env_int("AUDIO_OUTPUT_DEVICE_INDEX")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )
    )

    # --- Wake-word gate + STT bridge ------------------------------------------
    gate = WakeWordGate(
        wakeword=wakeword,
        threshold=wake_threshold,
        wake_listen_seconds=wake_listen_seconds,
        followup_seconds=followup_seconds,
    )
    # --- Speech-to-text: Speechmatics (server-side turn detection) -------------
    stt = SpeechmaticsSTTService(
        api_key=speechmatics_key,
        settings=SpeechmaticsSTTService.Settings(
            language=Language.EN,
            turn_detection_mode=SpeechmaticsSTTService.TurnDetectionMode.ADAPTIVE,
            # Max pause that still counts as "thinking" rather than "done"
            # (seconds). ADAPTIVE mode ends the turn sooner when the sentence
            # sounds complete, so a high value here mostly costs latency only
            # when you trail off mid-thought. Lower = snappier, but breathing
            # pauses may cut you off.
            end_of_utterance_silence_trigger=float(
                os.getenv("EOU_SILENCE_TRIGGER", "1.5")
            ),
        ),
    )
    stt_bridge = SttGateBridge(gate, stt)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input/output devices and exit (for configuring .env on new hardware).",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
    else:
        asyncio.run(main())
