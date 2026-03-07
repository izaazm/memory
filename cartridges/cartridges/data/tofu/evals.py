"""
TOFU QA evaluation dataset for cartridge capacity stress testing.

Evaluates on the same QA pairs used for training — since we're measuring
how much information the cartridge can hold (not generalization), eval = train.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import random

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING
from cartridges.data.tofu.utils import load_tofu_authors, TOFUAuthor
from cartridges.utils import get_logger

logger = get_logger(__name__)


class TOFUQAGenerateDataset(GenerateEvalDataset):
    """
    Generation evaluation dataset for TOFU QA pairs.
    
    Evaluates the cartridge's ability to answer questions about the stored authors
    using ROUGE-L scoring. Since we're measuring capacity (not generalization),
    the eval set is identical to the training set.
    """

    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        num_authors: int = 1
        seed: int = 42
        max_questions: Optional[int] = None
        author_offset: int = 0  # skip this many authors (for modular eval on author subsets)

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer

        authors = load_tofu_authors(
            num_authors=config.num_authors + config.author_offset,
            seed=config.seed,
        )
        # Slice to the requested subset (supports modular eval on author subsets)
        authors = authors[config.author_offset:]

        # Flatten all QA pairs into eval questions
        self.questions: List[dict] = []
        for author in authors:
            for i, qa in enumerate(author.qa_pairs):
                self.questions.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "author_index": author.index,
                    "qa_index": i,
                    "convo_id": f"author{author.index}_q{i}",
                })

        random.Random(seed).shuffle(self.questions)

        if self.config.max_questions is not None:
            self.questions = self.questions[: self.config.max_questions]

        self.convo_id_to_idx = {
            q["convo_id"]: idx for idx, q in enumerate(self.questions)
        }

        logger.info(
            f"TOFUQAGenerateDataset: {len(self.questions)} questions "
            f"from {config.num_authors} authors"
        )

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        question = self.questions[index]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.cot

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question["question"]}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=question["question"],
            answer=question["answer"],
            convo_id=question["convo_id"],
            metadata={
                "author_index": question["author_index"],
                "qa_index": question["qa_index"],
            },
        )

    def __len__(self) -> int:
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str,
    ) -> Tuple[float, Dict[str, Optional[str]]]:
        """
        Score a prediction against the ground truth answer using ROUGE-L.

        Returns:
            (rouge_l_score, metadata_dict)
        """
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(answer, pred)
        rouge_l = scores["rougeL"].fmeasure

        return rouge_l, {
            "rouge_l": rouge_l,
            "pred_preview": pred[:200] if pred else "",
            "answer_preview": answer[:200] if answer else "",
        }
