from typing import List, Optional
from pprint import pprint

import torch
from transformers import GPT2Tokenizer, GPT2Model

from db import Collections
from data_classes import NewsArticle


class ArticleFinder:
    def __init__(
        self,
        model,
        tokenizer,
        articles: List[NewsArticle],
        news_embedding: torch.Tensor,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.articles = articles
        self.news_embeddings = news_embedding

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, index: int) -> NewsArticle:
        return self.articles[index]

    def get_articles_by_page_number(
        self, page_number: int, page_size: int
    ) -> List[NewsArticle]:
        start = (page_number - 1) * page_size
        end = start + page_size
        return self.articles[start : min(end, len(self.articles))]

    def get_article_by_index(self, index: int) -> Optional[NewsArticle]:
        if index >= len(self.articles):
            raise "Index out of range"

        return self.articles[index]

    def get_one_to_one_cos_sim(self, article_index: int) -> torch.Tensor:
        return torch.cosine_similarity(
            self.news_embeddings[article_index], self.news_embeddings, dim=2
        ).mean(1)

    def get_one_to_many_cos_sim(self, article_index: int) -> torch.Tensor:
        article_emb = self.news_embeddings[article_index] # size: (n_summary, embedding_size)
        results = [torch.cosine_similarity(emb.unsqueeze(0), news_emb, dim=2).mean(1) for emb in article_emb] # size: (n_summary, n_news)
        return torch.stack(results).mean(0) # size: (n_news)

    def find_n_related_articles(self, article_index: int, n: int, one_to_one: bool = True) -> List[NewsArticle]:
        if one_to_one:
            cos_sim = self.get_one_to_one_cos_sim(article_index)
        else:
            cos_sim = self.get_one_to_many_cos_sim(article_index)
        _, indices = torch.sort(cos_sim, descending=True)
        indices = indices[: n + 1].tolist()
        indices.remove(article_index)
        return [self.articles[index] for index in indices]

    def sentence_to_vector(self, sent: str) -> torch.tensor:
        token = self.tokenizer.encode(sent)
        tensor = torch.tensor([token])
        with torch.no_grad():
            # size: (1, embedding_size)
            return self.model(tensor).last_hidden_state.mean(dim=1)

    def search_n_related_articles(self, search_text: str, n: int) -> List[NewsArticle]:
        search_emb = self.sentence_to_vector(search_text)
        cos_sim = torch.cosine_similarity(search_emb, self.news_embeddings, dim=2).mean(
            1
        )
        _, indices = torch.sort(cos_sim, descending=True)
        indices = indices[:n].tolist()
        return [self.articles[index] for index in indices]


if __name__ == "__main__":
    collection = Collections().article_collection
    model = GPT2Model.from_pretrained("gpt2-xl")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    articles = collection.find()
    articles = [NewsArticle.from_json(i, article) for i, article in enumerate(articles)]
    news_emb: torch.Tensor = torch.load("data/news_emb.pt", map_location=torch.device("cpu"))


    finder = ArticleFinder(model, tokenizer, articles, news_emb)
    a = finder.find_n_related_articles(0, 10)
    pprint(a)
    b = finder.find_n_related_articles(0, 10, one_to_one=False)
    pprint(b)
