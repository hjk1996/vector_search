from typing import List

import torch
import torch.nn as nn
from bible import Bible

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util, losses





class ModelWrapper:
    def __init__(
        self, model, tokenizer, bible: Bible, embedding: torch.Tensor, name: str, device
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.bible = bible
        self.embedding = embedding
        self.name = name
        self.device = device

    def sentence_to_vector(self, sent: str) -> torch.tensor:
        token = self.tokenizer.encode(sent)
        tensor = torch.tensor([token]).to(self.device)
        with torch.no_grad():
            # size: (1, embedding_size)
            return self.model(tensor).last_hidden_state.mean(dim=1)

    def get_cos_sim(self, sentence: str) -> torch.tensor:
        sent_emb = self.sentence_to_vector(sentence)
        emb_size = self.embedding.size()
        # size: (chapter_size, verse_size)
        cos_sim = torch.nn.functional.cosine_similarity(sent_emb, self.embedding, dim=2)
        if emb_size[1] == 176:
            n_nonzero = torch.count_nonzero(torch.sum(self.embedding, dim=2), dim=1)
            # size: (chapter_size)
            return cos_sim.sum(1) / n_nonzero
        else:
            # size: (chapter_size)
            return cos_sim.mean(1)

    def get_top_n_results(self, sims: torch.tensor, n: int) -> List[int]:
        values, indices = torch.sort(sims, descending=True)
        values, indices = values[:n].tolist(), indices[:n].tolist()
        return values, indices

    def get_related_n_chapters(self, sentence: str, n: int) -> List[dict]:
        sims = self.get_cos_sim(sentence)
        _, indices = self.get_top_n_results(sims, n)

        chapters = [self.bible.get_chapter_name_by_index(index) for index in indices]
        elements = [chapter.split(" ") for chapter in chapters]
        books = [" ".join(element[:-1]) for element in elements]
        numbers = [int(element[-1]) for element in elements]
        chpter_texts = [
            self.bible.get_chapter_text(book, number)
            for book, number in zip(books, numbers)
        ]
        return [
            {"chapter_name": chapter, "verses": text}
            for chapter, text in zip(chapters, chpter_texts)
        ]

    def get_top_n_acc(self, gts: dict, n: int) -> float:
        total = 0
        correct = 0

        chapter_names = list(gts.keys())
        chapter_indices = [
            self.bible.get_chapter_indices(chapter_name)
            for chapter_name in chapter_names
        ]
        sentences = list(gts.values())

        for gt, sent in zip(chapter_indices, sentences):
            sims = self.get_cos_sim(sent)
            values, indices = self.get_top_n_results(sims, n)
            gt, indices = set(gt), set(indices)
            if len(gt.intersection(indices)) > 0:
                correct += 1
            total += 1

        return correct / total


class Ensemble:
    def __init__(self, model_wrappers: List[ModelWrapper]):
        self.model_wrappers = model_wrappers
        self.name = f"Ensemble of {[model.name for model in model_wrappers]} models"

    def get_cos_sim(self, sentence: str) -> torch.tensor:
        sims = []
        for model_wrapper in self.model_wrappers:
            sims.append(model_wrapper.get_cos_sim(sentence))
        return torch.stack(sims).mean(dim=0)

    def get_top_n_results(self, sims: torch.tensor, n: int) -> List[int]:
        values, indices = torch.sort(sims, descending=True)
        values, indices = values[:n].tolist(), indices[:n].tolist()
        return values, indices

    def get_top_n_acc(self, bible: Bible, gts: dict, n: int) -> float:
        total = 0
        correct = 0

        chapter_names = list(gts.keys())
        chapter_indices = [
            bible.get_chapter_indices(chapter_name) for chapter_name in chapter_names
        ]
        sentences = list(gts.values())

        for gt, sent in zip(chapter_indices, sentences):
            sims = self.get_cos_sim(sent)
            values, indices = self.get_top_n_results(sims, n)
            gt, indices = set(gt), set(indices)
            if len(gt.intersection(indices)) > 0:
                correct += 1
            total += 1

        return correct / total


class MyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.st = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.book_linear = nn.Linear(384, 176)
        self.chapter_linear = nn.Linear(384, 1189)


    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, sentences):
        encoded_input =  self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        t_out = self.st(**encoded_input) 
        s_embeddings = self.mean_pooling(t_out, encoded_input['attention_mask'])
        book_logits = self.book_linear(s_embeddings)
        chapter_logits = self.chapter_linear(s_embeddings)
        return s_embeddings, book_logits, chapter_logits


if __name__ == "__main__":
    model = MyModel()

    s, b, c = model("Hello world")

    print(s.size())
    print(b.size())
    print(c.size())
