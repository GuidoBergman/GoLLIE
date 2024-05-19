from typing import List

from ..utils_typing import Entity, dataclass


"""Entity definitions

The entity definitions are derived from the official SemEval-2023 Task 3 Subtask 3 guidelines:
https://aclanthology.org/2023.semeval-1.317v1.pdf
"""


@dataclass
class AtackOnReputation(Entity):
    """{ner_attack_on_reputation}"""

    span: str  # {ner_attack_on_reputation_examples}


@dataclass
class ManipulativeWording(Entity):
    """{ner_manipulative_wording}"""

    span: str  # {ner_manipulative_wording_examples}





ENTITY_DEFINITIONS: List[Entity] = [
    AtackOnReputation,
    ManipulativeWording
]



# __all__ = list(map(str, [*ENTITY_DEFINITIONS]))
