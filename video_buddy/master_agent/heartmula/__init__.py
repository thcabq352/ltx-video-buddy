"""HeartMuLa pack — local lyrics+tags music and lyric transcription.

Soft-imports ``heartlib`` (https://github.com/HeartMuLa/heartlib, Apache-2.0).
Importing this package does not import heartlib, touch the network, or load weights.
"""

from master_agent.heartmula.generate import (
    HeartMuLaUnavailable,
    MissingHeartMuLaWeights,
    generate_track,
    plan_generate,
)
from master_agent.heartmula.transcribe import plan_transcribe, transcribe_audio

__all__ = [
    "HeartMuLaUnavailable",
    "MissingHeartMuLaWeights",
    "generate_track",
    "plan_generate",
    "plan_transcribe",
    "transcribe_audio",
]
