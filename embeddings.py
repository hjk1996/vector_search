import json
from typing import List

import torch
from data_classes import ChapterSummary
from tqdm import tqdm
from glob import glob
import transformers

from bible import Bible
from data_classes import ChapterSummary, NewsArticle


def load_embeddings(embdding_folder: str, device: torch.device) -> dict:
    files = glob(f"{embdding_folder}/*.pt")
    embeddings = {}
    for file in files:
        emb = torch.load(file, map_location=device)
        embeddings[file.split("/")[-1].split(".")[0]] = emb
    return embeddings


def sentence_to_vector(model, tokenizer, device, sent: str) -> torch.Tensor:
    token = tokenizer.encode(sent)
    tensor = torch.tensor([token]).to(device)
    with torch.no_grad():
        embedding = model(tensor).last_hidden_state.mean(dim=1)
    return embedding


def make_bible_embedding(
    model, tokenizer, device: torch.device, bible: Bible, emb_size: int
) -> torch.Tensor:
    bible_emb = []
    for book in bible.books:
        for chapter in tqdm(book.chapters):
            chapter_emb = []
            for verse in chapter.verses:
                emb = sentence_to_vector(model, tokenizer, device, verse.text).to(device)
                chapter_emb.append(emb)
            chapter_emb = torch.cat(chapter_emb, dim=0)
            if chapter_emb.shape[0] < 176:
                chapter_emb = torch.cat(
                    [chapter_emb, torch.zeros((176 - chapter_emb.shape[0], emb_size)).to(device)],
                    dim=0,
                )
            bible_emb.append(chapter_emb)
    return torch.stack(bible_emb, dim=0)


def make_basic_bible_summary_embedding(
    model, tokenizer, device, bible_summary: List[ChapterSummary]
) -> torch.Tensor:
    bible_emb = []
    for css in tqdm(bible_summary):
        ch_embs = []
        for cs in css.ten_line_summaries:
            token = tokenizer.encode(cs.summary)
            tensor = torch.tensor([token]).to(device)
            with torch.no_grad():
                out = model(tensor)
                emb = out.last_hidden_state.mean(dim=1)
                ch_embs.append(emb)
        ch_embs = torch.cat(ch_embs, dim=0)
        bible_emb.append(ch_embs)

    return torch.stack(bible_emb)

def make_news_embedding(
        model, tokenizer, device, articles: List[NewsArticle]
) -> torch.Tensor:
    news_embs = []
    for article in tqdm(articles):
        article_emb = []
        for summary in article.summaries:
            token = tokenizer.encode(summary)
            tensor = torch.tensor([token]).to(device)
            with torch.no_grad():
                out = model(tensor)
                emb = out.last_hidden_state.mean(dim=1)
                news_embs.append(emb)
        article_emb = torch.cat(article_emb, dim=0)
        news_embs.append(article_emb)
    return torch.stack(news_embs, dim=0)
            

if __name__ == "__main__":

    bible = Bible("data/nrsv_bible.xml", "data/chapter_index_map.json")
    with open("data/bible_summary.json", "r") as f:
        summaries = json.load(f)
    summaries = [ChapterSummary.from_json(s) for s in summaries]
    device = torch.device("mps")

    gpt2_xl = transformers.GPT2Model.from_pretrained("gpt2-xl").to(device)
    gpt2_xl_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2-xl")

    gpt_xl_embeddings = make_basic_bible_summary_embedding(
        gpt2_xl, gpt2_xl_tokenizer, device, summaries
    )

    torch.save(gpt_xl_embeddings, "embeddings/gpt_xl_embeddings.pt")
