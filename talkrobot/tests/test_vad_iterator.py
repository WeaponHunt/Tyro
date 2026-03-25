import sys
import os
import types
import numpy as np
# allow running this test directly: add repo root so `talkrobot` package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Provide lightweight stand-ins for heavy runtime deps so we can import the module in tests
import types as _types
for _m in ('sounddevice', 'pynput', 'loguru'):
    if _m not in sys.modules:
        sys.modules[_m] = _types.SimpleNamespace()

# make pynput.keyboard.Listener available
if 'pynput' in sys.modules:
    class _DummyListener:
        def __init__(self, on_press=None, on_release=None):
            pass
        def start(self):
            pass
        def stop(self):
            pass
    sys.modules['pynput'].keyboard = _types.SimpleNamespace(Listener=_DummyListener)

# fake loguru.logger
if 'loguru' in sys.modules:
    class _FakeLogger:
        def info(self, *a, **k):
            pass
        def debug(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass
        def error(self, *a, **k):
            pass
    sys.modules['loguru'].logger = _FakeLogger()

from talkrobot.core.audio_recorder import VADIterator


def make_fake_modules():
    # Fake torch with from_numpy identity
    fake_torch = types.SimpleNamespace()
    fake_torch.from_numpy = lambda x: x

    # Fake silero_vad with simple get_speech_timestamps: returns a start when any non-zero sample
    def get_speech_timestamps(tensor, model, sampling_rate=16000, return_seconds=False):
        # tensor is a numpy array in our fake
        if np.any(tensor != 0):
            # return a dict with 'start' at sample 0
            return [{'start': 0}]
        return []

    fake_silero = types.SimpleNamespace(get_speech_timestamps=get_speech_timestamps, load_silero_vad=lambda: None)

    return fake_torch, fake_silero


def test_vad_iterator_start_and_end():
    fake_torch, fake_silero = make_fake_modules()
    sys.modules['torch'] = fake_torch
    sys.modules['silero_vad'] = fake_silero

    # small sampling rate to make silence threshold small
    sr = 10
    silence_dur = 0.2  # threshold = 2 samples
    vad = VADIterator(model=None, sampling_rate=sr, silence_duration=silence_dur)

    # simulate a chunk with speech
    chunk_speech = np.array([0.1, 0.2, 0.0, 0.0], dtype=np.float32)
    events = vad(chunk_speech, return_seconds=False)
    assert any('start' in e for e in events), f"expected start event, got {events}"
    assert vad.in_speech

    # now simulate silent chunks; since threshold is small, after a couple calls should emit end
    chunk_silence = np.zeros(3, dtype=np.float32)
    events2 = vad(chunk_silence)
    # likely no immediate end until threshold reached
    if events2:
        assert any('end' in e for e in events2)
    else:
        # call again to trigger end
        events3 = vad(chunk_silence)
        assert any('end' in e for e in events3), f"expected end event, got {events3}"


if __name__ == '__main__':
    test_vad_iterator_start_and_end()
    print('ok')
