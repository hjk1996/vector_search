import argparse

import torch
import torch.multiprocessing as mp
from transformers import AdamW,     get_linear_schedule_with_warmup

import torch.nn.functional as F


from bible import Bible
from models import MyModel
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn



class Dataset:

    def __init__(self, bible: Bible):
        self.bible = bible
        self.make_pairs()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int):
        return self.pairs[index]

    def make_pairs(self):
        pairs = []
        for book in self.bible:
            for chapter in book:
                for verse in chapter:
                    for other_verse in verse.get_other_verse():
                        book_index_one_hot = F.one_hot(torch.tensor(book.book_index), num_classes=66)
                        chapter_index_one_hot = F.one_hot(torch.tensor(chapter.chapter_index), num_classes=1189)
                        pairs.append((verse.text, other_verse.text, verse.get_random_neg_pair(), book_index_one_hot, chapter_index_one_hot))
        
        self.pairs = pairs


def train_function(args):
    model = MyModel()
    optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.steps,
    )

    cross_entropy_loss = nn.CrossEntropyLoss()
    max_grad_norm = 1

    model.train()

    for step, batch in enumerate(dataloader):

        anchor = model.tokenizer(batch[0], padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
        positive = model.tokenizer(batch[1], padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
        negative = model.tokenizer(batch[2], padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")

        embeddings_a = model(**anchor.to(device))
        embeddings_p = model(**positive.to(device))
        embeddings_n = model(**negative.to(device))

        embeddings_c = torch.cat([embeddings_p, embeddings_n])

        scores = torch.matmul(embeddings_a, embeddings_c.T) * args.scale



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--nprocs', type=int, default=8)
    parser.add_argument('--scale', type=float, default=20, help="Use 20 for cossim, and 1 when you work with unnormalized embeddings with dot product")
    args = parser.parse_args()

    bible = Bible("./data/nrsv_bible.xml", "./data/chapter_index_map.json", "data/bible_summary_similarities.pt")

    dataset = Dataset(bible)
    dataloader= DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nprocs)

    

    # for batch in dataloader:
    #     anchors, positives, negatives, book_index_one_hot, chapter_index_one_hot = batch
    #     print(anchors)
    #     print(positives)
    #     print(negatives)
    #     print(book_index_one_hot)
    #     print(chapter_index_one_hot)
    #     break