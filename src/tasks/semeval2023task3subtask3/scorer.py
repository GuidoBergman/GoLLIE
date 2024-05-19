from typing import Dict, List, Type

from src.tasks.semeval2023task3subtask3.prompts import ENTITY_DEFINITIONS
from src.tasks.utils_scorer import SpanScorer
from src.tasks.utils_typing import Entity


class SemEvalEntityScorer(SpanScorer):
    """SemEval-2023 Task 3 Subtask 3 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS

    def __call__(self, reference: List[Entity], predictions: List[Entity]) -> Dict[str, Dict[str, float]]:
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}



