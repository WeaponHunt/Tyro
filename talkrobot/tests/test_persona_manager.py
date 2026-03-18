"""
PersonaManager 测试
"""
import json
import importlib.util
from pathlib import Path


def _load_persona_manager_class():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "talkrobot" / "core" / "persona_manager.py"
    spec = importlib.util.spec_from_file_location("persona_manager_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.PersonaManager


PersonaManager = _load_persona_manager_class()


def test_persona_manager_user_prompt(tmp_path):
    profile_path = tmp_path / "persona_profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "default": {"system_prompt": "default-prompt"},
                "alice": {"system_prompt": "alice-prompt"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manager = PersonaManager(str(profile_path), fallback_prompt="fallback-prompt")
    assert manager.get_prompt_for_user("alice") == "alice-prompt"


def test_persona_manager_default_fallback(tmp_path):
    profile_path = tmp_path / "persona_profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "default": {"system_prompt": "default-prompt"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manager = PersonaManager(str(profile_path), fallback_prompt="fallback-prompt")
    assert manager.get_prompt_for_user("new_user") == "default-prompt"


def test_persona_manager_config_fallback_when_file_missing(tmp_path):
    missing_path = tmp_path / "not_exists.json"
    manager = PersonaManager(str(missing_path), fallback_prompt="fallback-prompt")
    assert manager.get_prompt_for_user("any") == "fallback-prompt"
