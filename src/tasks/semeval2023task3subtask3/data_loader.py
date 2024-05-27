from typing import Dict, List, Tuple, Type, Union
import re

from src.tasks.semeval2023task3subtask3.guidelines_gold import EXAMPLES, GUIDELINES
from src.tasks.semeval2023task3subtask3.prompts import (
    ENTITY_DEFINITIONS,
    AtackOnReputation,
    ManipulativeWording
)
from src.tasks.label_encoding import rewrite_labels

from ..utils_data import DatasetLoader, Sampler
from ..utils_typing import Entity
import os


MAX_ARTICLE_LENGTH = 7500

def get_spans(labels_file):
    spans = []
    attack_on_reputation = ['Appeal_to_Hypocrisy', 'Guilt_by_Association',  'Name_Calling-Labeling',  'Questioning_the_Reputation', 'Doubt']
    manipulative_wordding = ['Exaggeration-Minimisation',  'Obfuscation-Vagueness-Confusion', 'Repetition', 'Loaded_Language']


    for line in labels_file:
       parts = line.strip().split('\t')
       label = parts[1]
       if label in manipulative_wordding:
          label = 'Manipulative_Wordding'
       elif label in attack_on_reputation:
          label = 'Attack_on_Reputation'
       else:
          continue

       start = int(parts[2])
       end = int(parts[3])

       if end >= MAX_ARTICLE_LENGTH:
           continue

       spans.append([label, start, end])

    return spans

def get_semeval(
    path: str,
    ENTITY_TO_CLASS_MAPPING: Dict[str, Type[Entity]],
) -> Tuple[List[List[str]], List[List[Entity]]]:
    """
    Get the semeval dataset
    Args:
        path (str): the path of the with the articles and the labels.
    Returns:
        (List[str],List[Union[AtackOnReputation,ManipulativeWording]]): The text and the entities
    """
    article_folder = os.path.join(path, 'articles')
    label_folder  = os.path.join(path, 'labels')
    article_files = os.listdir(article_folder)


    dataset_sentences: List[List[str]] = []
    dataset_entities: List[List[Entity]] = []

    for article_file in article_files:
        article_id = article_file.split('.')[0].replace('article', '')
        # Manually removing problematic articles
        if int(article_id) in (2331, 2328, 2287):
            continue
        article_path = os.path.join(article_folder, article_file)
        label_path = os.path.join(label_folder, f'article{article_id}-labels-subtask-3.txt')
        
        with open(label_path, 'r') as labels_file:
            spans = get_spans(labels_file)


        with open(article_path, 'r') as article_file:
            article_content = article_file.read()
            article_content = article_content[:MAX_ARTICLE_LENGTH]

            # Get entities
            entities = []
            for label, start, end in spans:
                entities.append(ENTITY_TO_CLASS_MAPPING[label](span=article_content[start:end]))

            words = re.split(r'\s', article_content)

            dataset_sentences.append(words)
            dataset_entities.append(entities)

    return dataset_sentences, dataset_entities


class SemEvalDatasetLoader(DatasetLoader):
    """
    A `DatasetLoader` for the SemEval-2023 Task 3 Subtask 3 dataset.

    Args:
        path(`str`):
            The path of the split dataset

    Raises:
        `ValueError`:
            raised when a not defined value found.
    """

    ENTITY_TO_CLASS_MAPPING = None

    def __init__(self, path: str, **kwargs) -> None:
        self.ENTITY_TO_CLASS_MAPPING = (
            {
                'Attack_on_Reputation': AtackOnReputation,
                'Manipulative_Wordding': ManipulativeWording,
            }
        )

        self.elements = {}


        dataset_words, dataset_entities = get_semeval(
                path=path,
                ENTITY_TO_CLASS_MAPPING=self.ENTITY_TO_CLASS_MAPPING,
        )
      

        for id, (words, entities) in enumerate(zip(dataset_words, dataset_entities)):
            self.elements[id] = {
                "id": id,
                "doc_id": id,
                "text": " ".join(words),
                "entities": entities,
                "gold": entities,
            }


class SemEvalSampler(Sampler):
    """
    A data `Sampler` for the SemEval-2023 Task 3 Subtask 3 dataset.

    Args:
        dataset_loader (`SemEvalDatasetLoader`):
            The dataset loader that contains the data information.
        task (`str`, optional):
            The task to sample. It must be one of the following: NER, VER, RE, EE.
            Defaults to `None`.
        split (`str`, optional):
            The path_or_split to sample. It must be one of the following: "train", "dev" or
            "test". Depending on the path_or_split the sampling strategy differs. Defaults to
            `"train"`.
        parallel_instances (`Union[int, Tuple[int, int]]`, optional):
            The number of sentences sampled in parallel. Options:

                * **`int`**: The amount of elements that will be sampled in parallel.
                * **`tuple`**: The range of elements that will be sampled in parallel.

            Defaults to 1.
        max_guidelines (`int`, optional):
            The number of guidelines to append to the example at the same time. If `-1`
            is given then all the guidelines are appended. Defaults to `-1`.
        guideline_dropout (`float`, optional):
            The probability to dropout a guideline definition for the given example. This
            is only applied on training. Defaults to `0.0`.
        seed (`float`, optional):
            The seed to sample the examples. Defaults to `0`.
        prompt_template (`str`, optional):
            The path to the prompt template. Defaults to `"templates/prompt.txt"`.
        ensure_positives_on_train (bool, optional):
            Whether to ensure that the guidelines of annotated examples are not removed.
            Defaults to `False`.
        dataset_name (str, optional):
            The name of the dataset. Defaults to `None`.
        scorer (`str`, optional):
           The scorer class import string. Defaults to `None`.
        sample_only_gold_guidelines (`bool`, optional):
            Whether to sample only guidelines of present annotations. Defaults to `False`.
    """

    def __init__(
        self,
        dataset_loader: SemEvalDatasetLoader,
        task: str = None,
        split: str = "train",
        parallel_instances: Union[int, Tuple[int, int]] = 1,
        max_guidelines: int = -1,
        guideline_dropout: float = 0.0,
        seed: float = 0,
        prompt_template: str = "templates/prompt.txt",
        ensure_positives_on_train: bool = False,
        dataset_name: str = None,
        scorer: str = None,
        sample_only_gold_guidelines: bool = False,
        **kwargs,
    ) -> None:
        assert task in [
            "NER",
        ], f"SemEval-2023 Task 3 Subtask 3 only supports NER task. {task} is not supported."

        task_definitions, task_target = {
            "NER": (ENTITY_DEFINITIONS, "entities"),
        }[task]

        super().__init__(
            dataset_loader=dataset_loader,
            task=task,
            split=split,
            parallel_instances=parallel_instances,
            max_guidelines=max_guidelines,
            guideline_dropout=guideline_dropout,
            seed=seed,
            prompt_template=prompt_template,
            ensure_positives_on_train=ensure_positives_on_train,
            sample_only_gold_guidelines=sample_only_gold_guidelines,
            dataset_name=dataset_name,
            scorer=scorer,
            task_definitions=task_definitions,
            task_target=task_target,
            definitions=GUIDELINES,
            examples=EXAMPLES,
            **kwargs,
        )
