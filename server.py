import torch
from fastapi import FastAPI
from transformers import GPT2Tokenizer, GPT2Model
from pydantic import BaseModel

from bible import Bible
from embeddings import load_embeddings
from models import ModelWrapper
from db import Collections
from data_classes import NewsArticle
from news_articles import ArticleFinder


tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2Model.from_pretrained("gpt2-xl")
bible = Bible("data/nrsv_bible.xml", "data/chapter_index_map.json")


bible = Bible("data/nrsv_bible.xml", "data/chapter_index_map.json")
device = torch.device("cpu")
embedding = torch.load("embeddings/gpt2_xl.pt", map_location=device)


gpt2_xl = GPT2Model.from_pretrained("gpt2-xl").to(device)
gpt2_xl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

gpt2_xl_model = ModelWrapper(
    model=gpt2_xl,
    tokenizer=gpt2_xl_tokenizer,
    bible=bible,
    embedding=embedding,
    name="gpt2-xl",
    device=device,
)


collection = Collections().article_collection
articles = collection.find()
articles = [NewsArticle.from_json(i, article) for i, article in enumerate(articles)]
news_emb = torch.load("data/news_emb.pt", map_location=device)
article_finder = ArticleFinder(model, tokenizer, articles, news_emb)


app = FastAPI()


class BibleQuery(BaseModel):
    query: str
    limit: int


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/bible_query")
async def bible_query(bible_query: BibleQuery):
    return gpt2_xl_model.get_related_n_chapters(bible_query.query, bible_query.limit)


@app.get("/news_articles/{page}")
async def news_articles(page: int):
    if 10 * (page - 1) > len(article_finder):
        return {"error": "Page does not exist", "data": None}
    elif page < 1:
        return {"error": "Page does not exist", "data": None}
    else:
        articles = article_finder[10 * (page - 1) : min(10 * page, len(article_finder))]
        return {"error": None, "data": [article.to_json() for article in articles]}


@app.get("/news_article/{article_index}")
async def news_article(article_index: int):
    article = article_finder.get_article_by_index(article_index)
    if article is str:
        return {"error": article, "data": None}
    else:
        related_articles = article_finder.find_n_related_articles(article_index, 10)
        return {
            "error": None,
            "data": {
                "article": article.to_json(),
                "related_articles": [article.to_json() for article in related_articles],
            },
        }
