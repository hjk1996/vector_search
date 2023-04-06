import argparse

import torch
import torch.multiprocessing as mp
from transformers import AdamW
import torch.nn.functional as F


from bible import Bible
from models import MyModel
from torch.utils.data import Dataset, DataLoader



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


def train_function():
    model = MyModel()
    optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)



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
    print(len(dataset))
    # dataloader= DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nprocs)

    # for batch in dataloader:
    #     anchors, positives, negatives, book_index_one_hot, chapter_index_one_hot = batch
    #     print(anchors)
    #     print(positives)
    #     print(negatives)
    #     print(book_index_one_hot)
    #     print(chapter_index_one_hot)
    #     break