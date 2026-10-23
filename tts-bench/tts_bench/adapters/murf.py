"""Murf Falcon 2 over its streaming websocket, ``/v1/speech/stream-input``.

Auth and the audio contract sit on the handshake query string (``api-key``,
``model``, ``sample_rate``, ``channel_type=MONO``, ``format=PCM``). A
``voice_config`` frame names the voice, locale and ``context_id`` before t0.
Text goes in as JSON frames on that context; ``end: true`` flushes and the
server answers with base64 PCM then ``final: true``. The socket stays open
after ``final``, so the receive loop must break on the flag.

Audio is requested as raw PCM (s16le, no header). ``format=WAV`` would prefix
the first chunk; a header that still arrives is stripped so the timeline holds
samples only.

There is no cancel. Closing the socket is the only way to stop in-flight
audio, so cancel is a declared exclusion rather than a measured one.

Reference: https://murf.ai/api/docs/websocket-api
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError

_RIFF = b"RIFF"
_WAV_HEADER_BYTES = 44


def pcm_from_chunk(chunk: bytes) -> bytes:
    """Raw PCM from one chunk: a leading WAV header dropped, anything else passed through."""
    if chunk.startswith(_RIFF) and len(chunk) > _WAV_HEADER_BYTES:
        data = chunk.find(b"data")
        return chunk[data + 8:] if data != -1 else chunk[_WAV_HEADER_BYTES:]
    return chunk


class MurfAdapter(WebSocketAdapter):
    name: ClassVar[str] = "murf"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 44100, 48000)
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, voice_config frame"
    base = "wss://global.api.murf.ai/v1/speech/stream-input"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?" + urlencode({
            "api-key": self.api_key,
            "model": self.config.model,
            "sample_rate": self.config.sample_rate,
            "channel_type": "MONO",
            "format": "PCM",
        })

    def _locale(self) -> str:
        locale = self.config.option("locale") or self.config.language or "en-US"
        return "en-US" if locale.lower() == "en" else locale

    def _context(self, context_id: str | None) -> str | None:
        return context_id or self._active

    async def _connect(self) -> None:
        try:
            await super()._connect()
        except AdapterError as exc:
            raise AdapterError(str(exc).replace(self.api_key, "<redacted>")) from exc.__cause__

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id
        await self._send_json({
            "voice_config": {"voiceId": self.config.voice, "locale": self._locale()},
            "context_id": context_id,
        })

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"text": text, "context_id": context_id, "end": False})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"text": "", "context_id": context_id, "end": True})

    def _on_message(self, message: dict[str, Any]) -> None:
        context_id = self._context(message.get("context_id"))
        error = message.get("error") or message.get("errorMessage")
        if error:
            self._on_error(context_id, str(error))
            return
        audio = message.get("audio")
        if audio and context_id:
            self._on_audio(context_id, pcm_from_chunk(base64.b64decode(audio)))
        if message.get("final") and context_id:
            self._on_done(context_id, trace_id=message.get("trace_id"))
