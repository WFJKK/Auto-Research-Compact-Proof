"""Agent backends: where a round's responses come from.

Every backend has `available(index)` and `respond(prompt, index)`, where index
counts the attempts this backend has answered in the run, so a resumed run
asks for the same response again. fake replays files; manual (Step 7) and api
(Step 8) come later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Response:
    text: str
    meta: dict = field(default_factory=dict)


class BackendError(RuntimeError):
    pass
