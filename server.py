import torch
from fastapi import FastAPI
from transformers import GPT2Tokenizer, GPT2Model
from pydantic import BaseModel

from bible import Bible
from embeddings import load_embeddings
from models import ModelWrapper

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

app = FastAPI()


class BibleQuery(BaseModel):
    query: str
    limit: int


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/bible_query")
async def get_related_n_chapters(bible_query: BibleQuery):
    return gpt2_xl_model.get_related_n_chapters(bible_query.query, bible_query.limit)
