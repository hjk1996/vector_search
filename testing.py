import json

import torch
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import openai
from transformers import pipeline
import transformers


import openai_functions as of
import utils
from bible import Bible
from db import BibleSummaryCollection
from embeddings import load_embeddings, make_bible_embedding



if __name__ == "__main__":
    bible = Bible("data/nrsv_bible.xml", "data/chapter_index_map.json")
    device = torch.device("mps")
    gpt2 = transformers.GPT2Model.from_pretrained('gpt2-large').to(device)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-large')
    
    emb = make_bible_embedding(gpt2, tokenizer,device, bible)
    torch.save(emb, "embeddings/gpt2_large_whole.pt")