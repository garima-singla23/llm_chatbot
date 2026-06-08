import pytest
from chatbot.memory import ConversationMemory, Turn


class TestConversationMemory:
    """Unit tests for ConversationMemory class."""

    def test_add_basic(self):
        """Test adding turns and retrieving them via to_messages()."""
        memory = ConversationMemory()
        
        memory.add("user", "Hello")
        memory.add("assistant", "Hi there!")
        memory.add("user", "How are you?")
        
        messages = memory.to_messages()
        
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    def test_deque_eviction_with_max_turns(self):
        """Test that deque evicts old turns when max_turns is exceeded."""
        memory = ConversationMemory(max_turns=2)
        
        # Adding 5 turns (max_turns=2 means deque maxlen=4, so 5th turn evicts the 1st)
        memory.add("user", "msg1")
        memory.add("assistant", "msg2")
        memory.add("user", "msg3")
        memory.add("assistant", "msg4")
        memory.add("user", "msg5")
        
        messages = memory.to_messages()
        
        # Only the last 4 turns should be kept (deque maxlen = 2 * max_turns = 4)
        assert len(messages) == 4
        assert messages[0] == {"role": "assistant", "content": "msg2"}
        assert messages[1] == {"role": "user", "content": "msg3"}
        assert messages[2] == {"role": "assistant", "content": "msg4"}
        assert messages[3] == {"role": "user", "content": "msg5"}

    def test_update_pref_single(self):
        """Test updating a single preference."""
        memory = ConversationMemory()
        
        memory.update_pref("seat", "window")
        
        assert memory.user_prefs == {"seat": "window"}

    def test_update_pref_overwrite(self):
        """Test that updating a preference key overwrites the old value."""
        memory = ConversationMemory()
        
        memory.update_pref("seat", "window")
        memory.update_pref("seat", "aisle")
        
        assert memory.user_prefs == {"seat": "aisle"}

    def test_update_pref_multiple(self):
        """Test updating multiple different preferences."""
        memory = ConversationMemory()
        
        memory.update_pref("seat", "window")
        memory.update_pref("meal", "vegan")
        memory.update_pref("airline", "indigo")
        
        assert memory.user_prefs == {
            "seat": "window",
            "meal": "vegan",
            "airline": "indigo"
        }

    def test_summary_with_prefs(self):
        """Test summary() format with preferences."""
        memory = ConversationMemory()
        
        memory.update_pref("seat", "window")
        summary = memory.summary()
        
        assert summary == "User preferences: seat=window"

    def test_summary_with_multiple_prefs(self):
        """Test summary() format with multiple preferences."""
        memory = ConversationMemory()
        
        memory.update_pref("seat", "window")
        memory.update_pref("meal", "vegan")
        summary = memory.summary()
        
        # Summary should contain both preferences
        assert "User preferences:" in summary
        assert "seat=window" in summary
        assert "meal=vegan" in summary

    def test_summary_empty(self):
        """Test that summary() returns empty string when no preferences set."""
        memory = ConversationMemory()
        
        summary = memory.summary()
        
        assert summary == ""

    def test_clear_removes_turns(self):
        """Test that clear() removes all turns."""
        memory = ConversationMemory()
        
        memory.add("user", "Hello")
        memory.add("assistant", "Hi")
        assert len(memory.to_messages()) == 2
        
        memory.clear()
        
        assert len(memory.to_messages()) == 0

    def test_clear_removes_active_booking(self):
        """Test that clear() removes active_booking."""
        memory = ConversationMemory()
        
        memory.active_booking["pnr"] = "ABC123"
        assert memory.active_booking == {"pnr": "ABC123"}
        
        memory.clear()
        
        assert memory.active_booking == {}

    def test_clear_preserves_prefs(self):
        """Test that clear() preserves user_prefs."""
        memory = ConversationMemory()
        
        memory.update_pref("seat", "window")
        memory.add("user", "Hello")
        memory.active_booking["pnr"] = "ABC123"
        
        memory.clear()
        
        # Turns and active_booking cleared, but prefs survive
        assert len(memory.to_messages()) == 0
        assert memory.active_booking == {}
        assert memory.user_prefs == {"seat": "window"}

    def test_to_messages_dict_structure(self):
        """Test that to_messages() returns dicts with exactly 'role' and 'content' keys."""
        memory = ConversationMemory()
        
        memory.add("user", "Hello")
        memory.add("assistant", "Hi")
        
        messages = memory.to_messages()
        
        for msg in messages:
            # Each dict should have exactly 2 keys: "role" and "content"
            assert set(msg.keys()) == {"role", "content"}
            assert isinstance(msg["role"], str)
            assert isinstance(msg["content"], str)

    def test_to_messages_preserves_order(self):
        """Test that to_messages() preserves the order of turns."""
        memory = ConversationMemory()
        
        for i in range(5):
            memory.add("user" if i % 2 == 0 else "assistant", f"message_{i}")
        
        messages = memory.to_messages()
        
        for i, msg in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role
            assert msg["content"] == f"message_{i}"
