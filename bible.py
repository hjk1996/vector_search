import random
import json
from typing import List

import torch
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET


from data_classes import TrainData

class Bible:
    def __init__(self, bible_file_path: str, chapter_index_map_path: str, similarities_path: str) -> None:
        self.root: Element = ET.parse(bible_file_path).getroot()
        self.chapter_index_map = self.load_chapter_index_map(chapter_index_map_path)
        self.chapter_summary_similarities = torch.load(similarities_path)

        self.books = list(map(lambda x: Book(self, x), self.root.findall("BIBLEBOOK")))
        self.book_names = {book.book_name.lower() for book in self.books}
        self.chapters: list[Chapter] = self.get_chapters()
        self.verses: list[Verse] = self.get_verses()

    def __iter__(self):
        return iter(self.books)
    
    def __len__(self):
        return len(self.books)
    
    def __getitem__(self, index: int):
        return self.books[index]

    def load_chapter_index_map(self, path: str) -> dict:
        with open(path, "r") as f:
            ch_map = json.load(f)
        
        return ch_map
    
    def get_chapters(self):
        chapters = []
        for book in self.books:
            for chapter in book:
                chapters.append(chapter)
        return chapters
    
    def get_verses(self):
        verses = []
        for chapter in self.chapters:
            for verse in chapter:
                verses.append(verse)
        return verses



    def get_chapter_indices(self, text: str) -> List[int]:
        text = text.strip().lower()
        words = text.split(" ")
        book_name = " ".join(words[:-1])
        chapter_number = words[-1]

        if book_name not in self.book_names:
            raise ValueError(f"Book name {book_name} not found in bible")
        elif not any(char.isdigit() for char in chapter_number):
            raise ValueError(f"Chapter number {chapter_number} is not a number")
        
        if "-" in chapter_number:
            start = int(chapter_number.split("-")[0])
            end = int(chapter_number.split("-")[1])
            chapter_numbers =  list(range(start, end + 1))
        else:
            chapter_numbers = [int(chapter_number)]
        
        chapter_names = [f"{book_name} {chapter_number}" for chapter_number in chapter_numbers]

        return [self.chapter_index_map[chapter_name] for chapter_name in chapter_names]



    def get_chapter_name_by_index(self, index: int) -> str:
        count = 0
        for book in self.books:
            for chapter in book:
                
                if count == index:
                    return f"{book.book_name} {chapter.chapter_number}"
                count += 1


    def get_book_name_by_index(self, index: int) -> str:
        return self.books[index].book_name

    def get_book_index_by_name(self, name: str) -> int:
        for i, book in enumerate(self.books):
            if book.book_name.lower() == name.lower():
                return i
        return -1

    def get_verse_text(self, book: any, chapter: int, verse: int) -> str:
        if type(book) == str:
            book: Book = self.books[self.get_book_index_by_name(book)]
        elif type(book) == int:
            book: Book = self.books[book]
        else:
            raise TypeError("book must be either a string or an integer")

        return  book[chapter - 1][verse - 1]

    def get_chapter_text(self, book: any, chapter: int) -> List[str]:
        if type(book) == str:
            book: Book = self.books[self.get_book_index_by_name(book)]
        elif type(book) == int:
            book: Book = self.books[book]
        else:
            raise TypeError("book must be either a string or an integer")
        
        return book[chapter - 1].get_text()


    def get_book_text(self, book: any) -> list[list[str]]:
        if type(book) == str:
            book: Book = self.books[self.get_book_index_by_name(book)]
        elif type(book) == int:
            book: Book = self.books[book]
        else:
            raise TypeError("book must be either a string or an integer")

        return book.get_text()
    

class Book:

    def __init__(self, bible:Bible, book: Element) -> None:
        self.book = book
        self.bible = bible
        self.book_number = int(book.attrib["bnumber"])
        self.book_index = self.book_number - 1
        self.book_name = book.attrib["bname"]
        self.chapters = list(map(lambda x: Chapter(self,x), self.book.findall("CHAPTER")))

    def __iter__(self):
        return iter(self.chapters)
    
    def __len__(self):
        return len(self.chapters)
    
    def __getitem__(self, index: int):
        return self.chapters[index]
    
    def __str__(self) -> str:
        return self.book_name
    
            
    def get_text(self) -> list[list[str]]:
        return list(map(lambda x: x.get_text(), self.chapters))

class Chapter:

    def __init__(self, book: Book,  chapter: Element) -> None:
        self.book = book
        self.chapter = chapter
        self.chapter_number = int(chapter.attrib["cnumber"])
        self.chapter_index = self.book.bible.chapter_index_map[f"{self.book.book_name} {self.chapter_number}"]
        self.verses = list(map(lambda x: Verse(self, x), self.chapter.findall("VERS")))

    
    def __len__(self):
        return len(self.verses)
    
    def __getitem__(self, index: int):
        return self.verses[index]
    
    def __str__(self) -> str:
        return f"{self.book.book_name} {self.chapter_number}"
    
    def get_text(self) -> list[str]:
        return list(map(lambda x: x.text, self.verses))

    def get_random_text(self) -> str:
        return random.choice(self.verses).text


class Verse:
    
        def __init__(self,chapter: Chapter, verse: Element) -> None:
            self.chapter = chapter
            self.verse = verse
            self.verse_number = int(verse.attrib["vnumber"])
            self.text = verse.text
    
        def __str__(self) -> str:
            return f"{self.chapter.book.book_name} {self.chapter.chapter_number}:{self.verse_number}"
        


        def get_other_verse(self) -> list:
            return [verse for verse in self.chapter if verse.verse_number != self.verse_number]

        def get_random_neg_pair(self, k: int = 5) -> str:
            sim = self.chapter.book.bible.chapter_summary_similarities[self.chapter.chapter_index]
            index =  random.choice(torch.topk(sim, k, largest=False).indices[0].tolist())
            return self.chapter.book.bible.chapters[index].get_random_text()
        

    
        



if __name__ == "__main__":
    bible = Bible("./data/nrsv_bible.xml", "./data/chapter_index_map.json")

    print(bible[-1][-1][-1].get_train_data())