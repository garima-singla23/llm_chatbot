from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Any
import time


@dataclass
class Turn:
	role: str
	content: str
	timestamp: float = field(default_factory=time.time)


class ConversationMemory:
	def __init__(self, max_turns: int = 10):
		self.turns: Deque[Turn] = deque(maxlen=max_turns * 2)
		self.user_prefs: Dict[str, Any] = {}
		self.active_booking: Dict[str, Any] = {}

	def add(self, role: str, content: str) -> None:
		"""Append a Turn to the conversation history."""
		self.turns.append(Turn(role=role, content=content))

	def update_pref(self, key: str, value: Any) -> None:
		"""Update a user preference key/value."""
		self.user_prefs[key] = value

	def to_messages(self) -> List[Dict[str, str]]:
		"""Return the conversation as a list of role/content dicts suitable for OpenAI API."""
		return [{"role": t.role, "content": t.content} for t in list(self.turns)]

	def summary(self) -> str:
		"""Return a short summary string of user preferences or empty string if none."""
		if not self.user_prefs:
			return ""
		pairs = [f"{k}={v}" for k, v in self.user_prefs.items()]
		return "User preferences: " + ", ".join(pairs)

	def clear(self) -> None:
		"""Clear conversation turns and active booking, keep user preferences."""
		self.turns.clear()
		self.active_booking.clear()


__all__ = ["Turn", "ConversationMemory"]

