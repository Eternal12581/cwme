"""Episode-local execution guard (blacklist / stagnation).

Moved out of PatternLibrary — cross-episode knowledge stays in
PatternLibrary; temporary execution state resets each episode.
"""
from __future__ import annotations


class EpisodeGuard:
    def __init__(self):
        self.blacklist: set[str] = set()
        self.stagnant_counts: dict[str, int] = {}
        self.state_action_counts: dict[tuple[str, str], int] = {}
        self.state_visits: dict[str, int] = {}

    def reset(self) -> None:
        self.blacklist.clear()
        self.stagnant_counts.clear()
        self.state_action_counts.clear()
        self.state_visits.clear()

    def note_action(self, action: str, *, progress: bool = False) -> None:
        act = (action or "").strip().lower()
        if not act:
            return
        if progress:
            self.stagnant_counts.clear()
            return
        self.stagnant_counts[act] = int(self.stagnant_counts.get(act, 0) or 0) + 1
        if self.stagnant_counts[act] >= 2:
            self.blacklist.add(act)

    def note_score_gain(self) -> None:
        self.stagnant_counts.clear()

    def note_transition(self, state_signature: str, action: str, *, progress: bool) -> None:
        """Block a failed/no-op action only in the state where it failed."""
        state = (state_signature or "").strip().lower()
        act = (action or "").strip().lower()
        if not state or not act:
            return
        self.state_visits[state] = int(self.state_visits.get(state, 0) or 0) + 1
        key = (state, act)
        if progress:
            self.state_action_counts.pop(key, None)
            return
        count = int(self.state_action_counts.get(key, 0) or 0) + 1
        self.state_action_counts[key] = count
        # One observed no-op/failure blocks the immediate repeat that causes
        # the most common navigation and fixture loops.
        if count >= 1:
            self.blacklist.add(f"stateact:{state}:{act}")

    def is_blocked(self, action: str, *, state_signature: str = "") -> bool:
        act = (action or "").strip().lower()
        if not act:
            return False
        if act in self.blacklist:
            return True
        state = (state_signature or "").strip().lower()
        return bool(state and f"stateact:{state}:{act}" in self.blacklist)

    def note_pattern_failure(self, pattern_id: str, milestone: str) -> None:
        pid = (pattern_id or "").strip()
        ms = (milestone or "").strip().lower()
        if pid:
            self.blacklist.add(f"pat:{pid}")
        if pid and ms:
            self.blacklist.add(f"pat:{pid}:{ms}")
        elif ms:
            self.blacklist.add(ms)

    def is_pattern_blocked(self, pattern_id: str, milestone: str) -> bool:
        ms = (milestone or "").strip().lower()
        pid = (pattern_id or "").strip()
        if pid and f"pat:{pid}" in self.blacklist:
            return True
        if pid and f"pat:{pid}:{ms}" in self.blacklist:
            return True
        return ms in self.blacklist if ms else False
