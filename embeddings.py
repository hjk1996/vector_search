import torch
from typing import List
from data_classes import ChapterSummary
from tqdm import tqdm
from glob import glob

def load_embeddings(embdding_folder: str) -> dict:
    files = glob(f"{embdding_folder}/*.pt")
    embeddings = {}
    for file in files:
        emb = torch.load(file)
        embeddings[file.split("/")[-1].split(".")[0]] = emb
    return embeddings
  

def sentence_to_vector(model, tokenizer, device, sent: str) -> torch.Tensor:
  token = tokenizer.encode(sent)
  tensor = torch.tensor([token]).to(device)
  with torch.no_grad():
    embedding = model(tensor).last_hidden_state.mean(dim=1)
  return embedding

def make_bible_embedding(model, tokenizer, device, bible_summary: List[ChapterSummary]) -> torch.Tensor:
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