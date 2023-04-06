import json

from sentence_transformers import SentenceTransformer, util, losses
from transformers import AutoTokenizer
import torch

from bible import Bible

if __name__ == "__main__":
    bible = Bible("./data/nrsv_bible.xml", "./data/chapter_index_map.json", "data/bible_summary_similarities.pt")
    print(bible[18].book_name)
    longest = " ". join(bible[18][118].get_text())
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    print(tokenizer.max_len_single_sentence)
    a =  tokenizer(longest)
    print(len(a["input_ids"]))
