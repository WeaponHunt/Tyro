import threading
import time
import importlib.util
from pathlib import Path
import sys
import types


if "loguru" not in sys.modules:
    _logger_stub = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["loguru"] = types.SimpleNamespace(logger=_logger_stub)

if "pynput" not in sys.modules:
    _keyboard_stub = types.SimpleNamespace(Listener=lambda *args, **kwargs: None)
    sys.modules["pynput"] = types.SimpleNamespace(keyboard=_keyboard_stub)


_MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "conversation_manager.py"
_SPEC = importlib.util.spec_from_file_location("conversation_manager_under_test", _MODULE_PATH)
cm_module = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(cm_module)
ConversationManager = cm_module.ConversationManager


class _DummyListener:
    def __init__(self, on_press=None, daemon=True):
        self.on_press = on_press
        self.daemon = daemon

    def start(self):
        return None

    def stop(self):
        return None


class _DummyASR:
    def transcribe(self, _audio):
        return ""


class _DummyTTS:
    def __init__(self):
        self.stop_calls = 0
        self._interrupted = threading.Event()

    def synthesize(self, _text, play_audio=True):
        return []

    def stop(self):
        self.stop_calls += 1


class _DummyLLM:
    def publish_gesture(self, _gesture):
        return True

    def publish_head_manual_control_deg(self, _x, _y):
        return True

    def publish_face_tracking_enable(self, _enabled):
        return True


class _DummyMemory:
    def add_memory(self, *_args, **_kwargs):
        return None

    def search_memory(self, *_args, **_kwargs):
        return ""


class _DummyAudioRecorder:
    listen_mode = "continuous"

    def __init__(self):
        self.is_tts_playing = False


def _build_manager(monkeypatch, **kwargs):
    monkeypatch.setattr(cm_module.keyboard, "Listener", _DummyListener)
    return ConversationManager(
        asr_module=_DummyASR(),
        tts_module=_DummyTTS(),
        llm_module=_DummyLLM(),
        memory_module=_DummyMemory(),
        tts_enabled=True,
        script_pause_resume_key="p",
        **kwargs,
    )


def test_split_text_for_script_step():
    segments = ConversationManager._split_text_for_script_step("a,b。c，d.")
    assert segments == ["a,", "b。", "c，", "d."]


def test_script_pause_resume_replays_current_substep(monkeypatch):
    manager = _build_manager(monkeypatch)

    played_segments = []
    pause_triggered = {"done": False}

    def _fake_play(segment_text):
        played_segments.append(segment_text)
        if not pause_triggered["done"]:
            pause_triggered["done"] = True
            manager._toggle_script_pause_mode()
            timer = threading.Timer(0.05, manager._toggle_script_pause_mode)
            timer.daemon = True
            timer.start()
            time.sleep(0.02)

    monkeypatch.setattr(
        manager,
        "_load_script_steps",
        lambda: [
            {
                "text": "第一句，第二句。",
                "expression": "happy",
                "image": "",
                "delay": 0.0,
            }
        ],
    )
    monkeypatch.setattr(manager, "_play_tts_notice", _fake_play)

    manager._script_mode = True
    manager._run_script_mode()

    # 第一个小step被暂停后，恢复会从同一个小step重新开始
    assert played_segments == ["第一句，", "第一句，", "第二句。"]


def test_resolve_script_path_uses_configured_file(tmp_path, monkeypatch):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    (script_dir / "a.json").write_text('{"name": "a", "steps": []}', encoding="utf-8")
    (script_dir / "target.json").write_text('{"name": "target", "steps": []}', encoding="utf-8")

    manager = _build_manager(
        monkeypatch,
        script_dir=str(script_dir),
        script_file="target.json",
    )

    assert manager._resolve_script_path() == str(script_dir / "target.json")


def test_no_face_timeout_exits_continuous_response_mode(monkeypatch):
    manager = _build_manager(
        monkeypatch,
        audio_recorder=_DummyAudioRecorder(),
        no_face_response_timeout_seconds=0.05,
    )
    manager._response_enabled = True

    manager.on_face_user_change("guest", is_familiar=False, has_face=False)
    time.sleep(0.12)

    assert manager._response_enabled is False
    manager.shutdown(timeout=0.1)


def test_no_face_timeout_starts_after_tts_finishes(monkeypatch):
    audio_recorder = _DummyAudioRecorder()
    manager = _build_manager(
        monkeypatch,
        audio_recorder=audio_recorder,
        no_face_response_timeout_seconds=0.05,
    )
    manager._response_enabled = True
    audio_recorder.is_tts_playing = True

    manager.on_face_user_change("guest", is_familiar=False, has_face=False)
    time.sleep(0.08)

    assert manager._response_enabled is True

    audio_recorder.is_tts_playing = False
    manager._mark_waiting_for_user_response()
    time.sleep(0.08)

    assert manager._response_enabled is False
    manager.shutdown(timeout=0.1)
