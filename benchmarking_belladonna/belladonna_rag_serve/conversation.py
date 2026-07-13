"""In-memory conversation store for the Belladonna chatbot.

Sessions are keyed by an opaque session id and hold a bounded list of
{role, content} turns. This is intentionally process-local and ephemeral:
the RAG service is single-node and conversations are short clinician
lookups, not durable records. If the process restarts, clients simply
start a fresh session (the API treats an unknown id as a new session).

Thread-safe: FastAPI/uvicorn serves requests from a worker pool, so every
mutation is guarded by a single lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List

# Keep only the most recent turns so prompts (and the condense step) stay
# bounded. One "turn" is one message; a Q+A is two messages.
MAX_MESSAGES = 16

# Drop sessions untouched for this long so memory doesn't grow unbounded.
SESSION_TTL_SECONDS = 6 * 60 * 60  # 6 hours


class _Session:
    __slots__ = ("messages", "last_seen")

    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []
        self.last_seen: float = time.time()


class ConversationStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.Lock()

    def _evict_expired_locked(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
        for sid in stale:
            del self._sessions[sid]

    def get_or_create(self, session_id: str | None) -> str:
        """Return a valid session id, creating one if needed (or if the
        supplied id is unknown, e.g. after a server restart)."""
        with self._lock:
            self._evict_expired_locked()
            if session_id and session_id in self._sessions:
                self._sessions[session_id].last_seen = time.time()
                return session_id
            new_id = session_id if session_id else uuid.uuid4().hex
            self._sessions[new_id] = _Session()
            return new_id

    def history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return list(session.messages)

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session()
                self._sessions[session_id] = session
            session.messages.append({"role": role, "content": content})
            # Trim oldest messages, keeping the tail bounded.
            if len(session.messages) > MAX_MESSAGES:
                session.messages = session.messages[-MAX_MESSAGES:]
            session.last_seen = time.time()

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# Module-level singleton; the whole service shares one store.
STORE = ConversationStore()
