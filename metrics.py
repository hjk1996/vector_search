from typing import List, Tuple

import torch

from embeddings import sentence_to_vector
from bible import Bible


def get_cos_sim(
    model, tokenizer, bible_embedding: torch.Tensor, sentence: str
) -> torch.Tensor:
    emb = sentence_to_vector(model, tokenizer, sentence)
    return torch.nn.functional.cosine_similarity(emb, bible_embedding, dim=2).mean(
        dim=1
    )


def get_top_n_results(sims: torch.Tensor, n: int) -> Tuple[List[float], List[int]]:
    values, indices = torch.sort(sims, descending=True)
    values, indices = values[:n].tolist(), indices[:n].tolist()
    return values, indices

def get_top_n_acc(model, tokenizer, bible: Bible, bible_embedding:torch.Tensor, gts: dict, n: int ) -> float:
    total = 0
    correct = 0

    chapter_names = list(gts.keys())
    chapter_indices = [bible.get_chapter_indices(chapter_name) for chapter_name in chapter_names]
    sentences = list(gts.values())

    for gt, sent in zip(chapter_indices, sentences):
        sims = get_cos_sim(model, tokenizer, bible_embedding, sent)
        values, indices = get_top_n_results(sims, n)
        gt, indices = set(gt), set(indices)
        if len(gt.intersection(indices)) > 0:
            correct += 1
        total += 1

    return correct / total


  