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
    packed_seq_length: int = 2048,
    corpus_text: Optional[str] = None,
) -> List[Conversation]:
    """
    Run the base model on TOFU conversations to extract top-k logprobs for each
    assistant token.  This produces the teacher logprobs needed for
    ``targets="logits"`` training (KL-divergence / soft-target distillation)
    instead of ``targets="tokens"`` (hard one-hot CE).

    When ``corpus_text`` is provided, a system message containing the corpus is
    prepended to each conversation before the forward pass.  This makes the
    extracted logprobs *context-conditioned* — i.e. the teacher answers **given
    the corpus in context** — which matches the self-study synthesis pipeline
    (bot B sees the corpus in its system prompt).  Without this, the logprobs
    only reflect the model's prior (question alone, no context) and are not
    useful for context distillation.

    Conversations are packed into fixed-length batches (using ``seq_ids`` to
    distinguish sequences) so that:
      1. Multiple conversations go through a single forward pass.
      2. The packed length is constant, avoiding triton recompilation.

    Args:
        conversations: TOFU conversations (user question + assistant answer).
        model: The FlexLlama / FlexQwen causal LM (already on *device*).
        tokenizer: The matching tokenizer.
        top_k: Number of top logprobs to keep per token position.
        device: CUDA device string.
        packed_seq_length: Fixed sequence length for each packed batch.
        corpus_text: If provided, prepended as a system message so that the
            teacher logprobs are conditioned on the corpus (context distillation).

    Returns:
        New list of Conversations with ``top_logprobs`` and ``token_ids``
        populated on the assistant message.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    from cartridges.clients.base import FlatTopLogprobs, TopLogprobs
    from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER

    converter = MODEL_TO_MESSAGE_CONVERTER[tokenizer.name_or_path.lower()]

    # --- Step 1: tokenize all conversations up front ---
    # If corpus_text is provided, prepend a system message so the teacher
    # logprobs are conditioned on the corpus (context distillation).
    system_message = (
        Conversation.Message(role="system", content=corpus_text, token_ids=None)
        if corpus_text is not None
        else None
    )

    elements = []
    for convo in conversations:
        messages = convo.messages
        if system_message is not None:
            messages = [system_message] + list(messages)
        elements.append(converter(messages, retokenize=True, tokenizer=tokenizer))

    # --- Step 2: pack conversations into fixed-length batches ---
    # Each batch packs as many conversations as fit in packed_seq_length.
    # This gives a single forward pass per batch and avoids recompilation.
    batches: list[list[int]] = []  # each batch is a list of conversation indices
    curr_batch: list[int] = []
    curr_len = 0
    for idx, elem in enumerate(elements):
        tok_len = len(elem.input_ids)
        if curr_len + tok_len > packed_seq_length and curr_batch:
            batches.append(curr_batch)
            curr_batch, curr_len = [], 0
        curr_batch.append(idx)
        curr_len += tok_len
    if curr_batch:
        batches.append(curr_batch)

    # --- Step 3: run batched forward passes ---
    # Store per-conversation results keyed by index
    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]] = {}

    for batch_idxs in tqdm(batches, desc="Rescoring conversations (batched)"):
        batch_input_ids, batch_seq_ids, batch_position_ids = [], [], []
        # Track where each conversation's targets land in the packed sequence
        offsets: list[int] = []  # cumulative token offset for each conv in the batch
        offset = 0
        for seq_id, conv_idx in enumerate(batch_idxs):
            elem = elements[conv_idx]
            n = len(elem.input_ids)
            batch_input_ids.append(elem.input_ids)
            batch_seq_ids.append(torch.full((n,), seq_id, dtype=torch.long))
            batch_position_ids.append(torch.arange(n, dtype=torch.long))
            offsets.append(offset)
            offset += n

        # Pad to fixed packed_seq_length to avoid triton recompilation
        total_tokens = offset
        pad_len = packed_seq_length - total_tokens
        if pad_len > 0:
            batch_input_ids.append(torch.zeros(pad_len, dtype=torch.long))
            batch_seq_ids.append(torch.full((pad_len,), len(batch_idxs), dtype=torch.long))
            batch_position_ids.append(torch.zeros(pad_len, dtype=torch.long))

        all_input_ids = torch.cat(batch_input_ids).to(device)
        all_seq_ids = torch.cat(batch_seq_ids).to(device)
        all_position_ids = torch.cat(batch_position_ids).to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=all_input_ids,
                    seq_ids=all_seq_ids,
                    position_ids=all_position_ids,
                )

        logits = outputs.logits[0]  # [packed_seq_length, vocab]

        # --- OOM optimisation -----------------------------------------------
        # Instead of materializing a full [packed_seq_length, vocab_size]
        # log-softmax tensor (which can be several GB for large vocabs), we
        # first gather only the target-prediction positions for each
        # conversation and compute log_softmax + topk on the much smaller
        # [total_target_positions, vocab] slice.
        # --------------------------------------------------------------------
        all_pred_positions = []
        conv_slices = {}  # conv_idx -> (start, end) into all_pred_positions
        cursor = 0
        for seq_id, conv_idx in enumerate(batch_idxs):
            elem = elements[conv_idx]
            target_abs = elem.topk_token_idxs + offsets[seq_id]
            pred_pos = target_abs - 1
            n = len(pred_pos)
            all_pred_positions.append(pred_pos)
            conv_slices[conv_idx] = (cursor, cursor + n)
            cursor += n

        all_pred_positions = torch.cat(all_pred_positions).to(device)
        target_logits = logits[all_pred_positions]        # [total_targets, vocab]
        del logits  # free the big tensor before log_softmax
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        del target_logits
        all_topk_vals, all_topk_ids = torch.topk(target_log_probs, k=top_k, dim=-1)
        del target_log_probs

        # Extract per-conversation logprobs
        for seq_id, conv_idx in enumerate(batch_idxs):
            start, end = conv_slices[conv_idx]
            topk_vals = all_topk_vals[start:end]
            topk_ids = all_topk_ids[start:end]

            num_targets = end - start

            # Build dense TopLogprobs and flatten with threshold pruning.
            # This matches the synthesis pipeline: only keep enough top-k entries
            # per position to cover 99% of probability mass. Without this, every
            # position has K=top_k entries, which makes loss.mean() K times smaller
            # than the tokens path and effectively kills the learning rate.
            dense_lp = TopLogprobs(
                logprobs=topk_vals.cpu().float().numpy().astype(np.float16),
                token_ids=topk_ids.cpu().numpy().astype(np.int32),
            )
            flat_lp = dense_lp.flatten(threshold=0.99)

            results[conv_idx] = (
                flat_lp.token_idx,
                flat_lp.token_id,
                flat_lp.logprobs,
                elem.topk_token_ids.numpy().tolist(),
                num_targets,
                flat_lp.shape,
            )

        del outputs, all_topk_vals, all_topk_ids

    # --- Step 4: assemble rescored conversations ---
    rescored: List[Conversation] = []
    for idx, convo in enumerate(conversations):
        token_idx, token_id, logprobs_arr, assistant_token_ids, num_targets, flat_shape = results[idx]

        flat_lp = FlatTopLogprobs(
            token_idx=token_idx,
            token_id=token_id,
            logprobs=logprobs_arr,
            shape=flat_shape,
        )

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

    logger.info(f"Rescored {len(rescored)} conversations in {len(batches)} batched forward passes (top-{top_k} logprobs)")
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
