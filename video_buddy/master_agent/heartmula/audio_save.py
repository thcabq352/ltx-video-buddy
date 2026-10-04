"""Write the generated waveform when torchaudio's torchcodec backend fails.

heartlib's ``postprocess`` calls ``torchaudio.save``. TorchAudio 2.9+ encodes
through torchcodec. Comfy's embedded Python often has a missing or broken
torchcodec DLL, so that call raises after HeartCodec has already decoded.

Buddy patches ``torchaudio.save`` for the generate call only. On failure the
same channels-first tensor is written with ``soundfile`` (a heartlib
dependency). Nothing in the Comfy environment is installed or upgraded.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable

log = logging.getLogger(__name__)


class HeartMuLaAudioSaveError(RuntimeError):
    """torchaudio and soundfile both failed to write the wav."""


def to_soundfile_array(audio: Any, *, channels_first: bool = True) -> Any:
    """``[channel, time]`` float tensor → soundfile ``[time, channel]`` array."""
    import numpy as np

    if hasattr(audio, "detach"):
        tensor = audio.detach()
        if hasattr(tensor, "float"):
            tensor = tensor.float()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        array = np.asarray(tensor.numpy() if hasattr(tensor, "numpy") else tensor)
    else:
        array = np.asarray(audio)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    if array.ndim != 2:
        raise HeartMuLaAudioSaveError(
            f"expected a 1D or 2D waveform, got shape {getattr(array, 'shape', None)}"
        )
    if channels_first:
        array = array.T
    return np.ascontiguousarray(array)


def soundfile_write(path: Any, array: Any, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(str(path), array, int(sample_rate))


def save_waveform(
    path: Any,
    src: Any,
    sample_rate: int,
    *,
    channels_first: bool = True,
    torchaudio_save: Callable[..., Any] | None = None,
) -> str:
    """Save one waveform. Returns ``torchaudio`` or ``soundfile``."""
    saver = torchaudio_save
    if saver is None:
        import torchaudio

        saver = torchaudio.save
    try:
        saver(path, src, int(sample_rate))
        return "torchaudio"
    except Exception as exc:
        try:
            array = to_soundfile_array(src, channels_first=channels_first)
            soundfile_write(path, array, int(sample_rate))
        except Exception as sf_exc:
            raise HeartMuLaAudioSaveError(
                "torchaudio.save failed and soundfile could not write the wav. "
                "Comfy's embedded Python often has a broken torchcodec DLL; "
                "Buddy falls back to soundfile without changing that env. "
                f"torchaudio: {exc}; soundfile: {sf_exc}"
            ) from sf_exc
        log.warning(
            "torchaudio.save failed (%s); wrote %s with soundfile", exc, path
        )
        return "soundfile"


@contextmanager
def waveform_save_fallback():
    """Route ``torchaudio.save`` through :func:`save_waveform` for one call.

    No-op when torchaudio is not installed. heartlib looks up
    ``torchaudio.save`` at call time, so patching the module attribute is
    enough. The original is restored on the way out.
    """
    try:
        import torchaudio
    except ImportError:
        yield
        return
    original = torchaudio.save

    def wrapped(uri: Any, src: Any, sample_rate: int, *args: Any, **kwargs: Any) -> str:
        channels_first = bool(kwargs.pop("channels_first", True))

        def _original(path: Any, data: Any, rate: int, **_ignored: Any) -> Any:
            return original(path, data, rate, *args, channels_first=channels_first, **kwargs)

        return save_waveform(
            uri,
            src,
            sample_rate,
            channels_first=channels_first,
            torchaudio_save=_original,
        )

    torchaudio.save = wrapped
    try:
        yield
    finally:
        torchaudio.save = original
