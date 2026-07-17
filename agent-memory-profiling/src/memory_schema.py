from dataclasses import dataclass
from typing import List


@dataclass
class MemoryRecord:
    memory_id: str
    user_id: str
    content: str
    memory_type: str
    module_tags: List[str]
    entity_tags: List[str]
    time: str
    importance: float
