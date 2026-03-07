"""
TOFU (Task of Fictitious Unlearning) dataset utilities.

The TOFU dataset contains 200 fictitious authors × 20 QA pairs each = 4000 rows total.
Authors are grouped sequentially: rows 0-19 = author 0, rows 20-39 = author 1, etc.

Dataset: https://huggingface.co/datasets/locuslab/TOFU
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import os
import tempfile

from cartridges.structs import Conversation
from cartridges.utils import get_logger

logger = get_logger(__name__)

QA_PER_AUTHOR = 20
TOTAL_AUTHORS = 200


@dataclass
class TOFUAuthor:
    """A single fictitious author from the TOFU dataset."""
    index: int  # 0-indexed author number
    qa_pairs: List[dict] = field(default_factory=list)  # list of {"question": ..., "answer": ...}

    @property
    def num_facts(self) -> int:
        return len(self.qa_pairs)

    def to_corpus_text(self) -> str:
        """Convert this author's QA pairs to a readable corpus for KV cache initialization."""
        lines = []
        for qa in self.qa_pairs:
            lines.append(f"{qa['answer']}")
        return " ".join(lines)

    def to_conversations(self) -> List[Conversation]:
        """Convert QA pairs into Conversation objects for training."""
        convos = []
        for i, qa in enumerate(self.qa_pairs):
            convo = Conversation(
                messages=[
                    Conversation.Message(
                        role="user",
                        content=qa["question"],
                        token_ids=None,
                    ),
                    Conversation.Message(
                        role="assistant",
                        content=qa["answer"],
                        token_ids=None,
                    ),
                ],
                system_prompt="",
                metadata={"author_index": self.index, "qa_index": i},
            )
            convos.append(convo)
        return convos


def load_tofu_authors(
    num_authors: int = TOTAL_AUTHORS,
    seed: int = 42,
) -> List[TOFUAuthor]:
    """
    Load the first `num_authors` authors from the TOFU dataset.
    
    Authors are deterministically ordered (rows 0-19 = author 0, etc.).
    The seed is reserved for future shuffling but currently authors are 
    loaded in the canonical order.
    
    Args:
        num_authors: Number of authors to load (1 to 200).
        seed: Random seed (reserved for future use).
    
    Returns:
        List of TOFUAuthor objects.
    """
    from datasets import load_dataset

    assert 1 <= num_authors <= TOTAL_AUTHORS, (
        f"num_authors must be between 1 and {TOTAL_AUTHORS}, got {num_authors}"
    )

    logger.info(f"Loading TOFU dataset with {num_authors} authors...")
    ds = load_dataset("locuslab/TOFU", "full", split="train")

    authors = []
    for author_idx in range(num_authors):
        start = author_idx * QA_PER_AUTHOR
        end = start + QA_PER_AUTHOR
        qa_pairs = [
            {"question": ds[i]["question"], "answer": ds[i]["answer"]}
            for i in range(start, end)
        ]
        authors.append(TOFUAuthor(index=author_idx, qa_pairs=qa_pairs))

    logger.info(
        f"Loaded {len(authors)} authors, "
        f"{sum(a.num_facts for a in authors)} total QA pairs."
    )
    return authors


def authors_to_conversations(authors: List[TOFUAuthor]) -> List[Conversation]:
    """Flatten a list of authors into a list of Conversation objects for training."""
    convos = []
    for author in authors:
        convos.extend(author.to_conversations())
    return convos


def authors_to_corpus_text(authors: List[TOFUAuthor]) -> str:
    """Concatenate all author QA pairs into a single corpus text string."""
    parts = []
    for author in authors:
        parts.append(f"--- Author {author.index} ---")
        parts.append(author.to_corpus_text())
    return "\n".join(parts)


def save_corpus_to_tempfile(authors: List[TOFUAuthor]) -> str:
    """
    Write the corpus text to a temporary file and return the path.
    Used for KVFromText initialization.
    """
    corpus = authors_to_corpus_text(authors)
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="tofu_corpus_")
    with os.fdopen(fd, "w") as f:
        f.write(corpus)
    logger.info(f"Saved TOFU corpus ({len(corpus)} chars) to {path}")
    return path


def rescore_tofu_conversations(
    conversations: List[Conversation],
    model,
    tokenizer,
    top_k: int = 20,
    device: str = "cuda",
) -> List[Conversation]:
    """
    Run the base model on TOFU conversations to extract top-k logprobs for each
    assistant token.  This produces the teacher logprobs needed for
    ``targets="logits"`` training (KL-divergence / soft-target distillation)
    instead of ``targets="tokens"`` (hard one-hot CE).

    Each TOFU conversation is assumed to have exactly one user turn followed by
    one assistant turn.

    Args:
        conversations: TOFU conversations (user question + assistant answer).
        model: The FlexLlama / FlexQwen causal LM (already on *device*).
        tokenizer: The matching tokenizer.
        top_k: Number of top logprobs to keep per token position.
        device: CUDA device string.

    Returns:
        New list of Conversations with ``top_logprobs`` and ``token_ids``
        populated on the assistant message.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    from cartridges.clients.base import FlatTopLogprobs
    from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER

    converter = MODEL_TO_MESSAGE_CONVERTER[tokenizer.name_or_path.lower()]

    rescored: List[Conversation] = []
    for convo in tqdm(conversations, desc="Rescoring conversations with model logprobs"):
        # Use the retokenize path to discover exact token positions and target IDs
        element = converter(convo.messages, retokenize=True, tokenizer=tokenizer)

        input_ids = element.input_ids.unsqueeze(0).to(device)
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                )

        logits = outputs.logits[0]                          # [seq_len, vocab]
        log_probs = F.log_softmax(logits, dim=-1)

        # Target positions are absolute indices in the full sequence.
        # The model's prediction for position P comes from logits[P-1].
        target_abs = element.topk_token_idxs                # [N_targets]
        pred_positions = target_abs - 1

        topk_vals, topk_ids = torch.topk(
            log_probs[pred_positions], k=top_k, dim=-1,
        )  # both [N_targets, top_k]

        num_targets = len(target_abs)

        flat_lp = FlatTopLogprobs(
            token_idx=np.repeat(np.arange(num_targets, dtype=np.int32), top_k),
            token_id=topk_ids.cpu().numpy().flatten().astype(np.int32),
            logprobs=topk_vals.cpu().float().numpy().flatten().astype(np.float16),
            shape=(num_targets, top_k),
        )

        # token_ids for the assistant message must include content + end tokens
        # (same set of tokens the retokenize path uses as targets)
        assistant_token_ids = element.topk_token_ids.numpy().tolist()

        new_messages = []
        for msg in convo.messages:
            if msg.role == "assistant":
                new_messages.append(
                    Conversation.Message(
                        role="assistant",
                        content=msg.content,
                        token_ids=assistant_token_ids,
                        top_logprobs=flat_lp,
                    )
                )
            else:
                new_messages.append(
                    Conversation.Message(
                        role=msg.role,
                        content=msg.content,
                        token_ids=None,
                        top_logprobs=None,
                    )
                )

        rescored.append(
            Conversation(
                messages=new_messages,
                system_prompt=convo.system_prompt,
                metadata=convo.metadata,
            )
        )

    logger.info(f"Rescored {len(rescored)} conversations with top-{top_k} logprobs")
    return rescored


def save_conversations_to_parquet(
    conversations: List[Conversation],
    path: str,
) -> str:
    """Save conversations to a parquet file for use with the cartridges training pipeline."""
    from cartridges.structs import write_conversations
    write_conversations(conversations, path)
    logger.info(f"Saved {len(conversations)} conversations to {path}")
    return path
