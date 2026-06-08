from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROFILE_PATH = Path("data/user_profile.json")


def load_profile() -> Dict[str, Any]:
  if not PROFILE_PATH.exists():
    return {}
  try:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
  except Exception:
    return {}


def save_profile(profile: Dict[str, Any]) -> None:
  PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
  PROFILE_PATH.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")


def update_profile(**fields: Any) -> Dict[str, Any]:
  profile = load_profile()
  profile.update({k: v for k, v in fields.items() if v is not None})
  save_profile(profile)
  return profile
