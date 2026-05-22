from talkrobot.core.dialogue_history import SlidingWindowDialogueHistory


def test_sliding_window_keeps_latest_rounds():
    history = SlidingWindowDialogueHistory(max_rounds=2)

    history.append("alice", "u1", "a1")
    history.append("alice", "u2", "a2")
    history.append("alice", "u3", "a3")

    context = history.build_context("alice")

    assert "u1" not in context
    assert "a1" not in context
    assert "u2" in context
    assert "a2" in context
    assert "u3" in context
    assert "a3" in context


def test_sliding_window_isolated_by_user():
    history = SlidingWindowDialogueHistory(max_rounds=2)

    history.append("alice", "alice-user", "alice-assistant")
    history.append("bob", "bob-user", "bob-assistant")

    alice_context = history.build_context("alice")
    bob_context = history.build_context("bob")

    assert "alice-user" in alice_context
    assert "bob-user" not in alice_context
    assert "bob-user" in bob_context
    assert "alice-user" not in bob_context


def test_clear_single_user_history():
    history = SlidingWindowDialogueHistory(max_rounds=2)

    history.append("alice", "alice-user", "alice-assistant")
    history.append("bob", "bob-user", "bob-assistant")
    history.clear("alice")

    assert history.build_context("alice") == ""
    assert "bob-user" in history.build_context("bob")
