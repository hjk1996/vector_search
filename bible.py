from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET


class Bible:
    def __init__(self, file_path: str) -> None:
        self.root: Element = ET.parse(file_path).getroot()
        self.books = list(map(lambda x: Book(x), self.root.findall("BIBLEBOOK")))

    def __iter__(self):
        return iter(self.books)
    
    def __len__(self):
        return len(self.books)
    
    def __getitem__(self, index: int):
        return self.books[index]

    def get_chapter_name_by_index(self, index: int) -> str:
        count = 0
        for book in self.books:
            for chapter in book:
                
                if count == index:
                    return f"{book.name} {chapter.number}"
                count += 1


    def get_book_name_by_index(self, index: int) -> str:
        return self.books[index].name

    def get_book_index_by_name(self, name: str) -> int:
        for i, book in enumerate(self.books):
            if book.name.lower() == name.lower():
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

    def get_chapter_text(self, book: any, chapter: int) -> list[str]:
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

    def __init__(self, book: Element) -> None:
        self.book = book
        self.number = int(book.attrib["bnumber"])
        self.name = book.attrib["bname"]
        self.chapters = list(map(lambda x: Chapter(self.name, x), self.book.findall("CHAPTER")))

    def __iter__(self):
        return iter(self.chapters)
    
    def __len__(self):
        return len(self.chapters)
    
    def __getitem__(self, index: int):
        return self.chapters[index]
    
    def __str__(self) -> str:
        return self.name
    
    def get_text(self) -> list[list[str]]:
        return list(map(lambda x: x.get_text(), self.chapters))

class Chapter:

    def __init__(self, book_name:str, chapter: Element) -> None:
        self.chapter = chapter
        self.book_name = book_name
        self.number = int(chapter.attrib["cnumber"])
        self.verses = list(map(lambda x: Verse(self.book_name, self.number, x), self.chapter.findall("VERS")))

    def __iter__(self):
        return iter(self.verses)
    
    def __len__(self):
        return len(self.verses)
    
    def __getitem__(self, index: int):
        return self.verses[index]
    
    def __str__(self) -> str:
        return f"{self.book_name} {self.number}"
    
    def get_text(self) -> list[str]:
        return list(map(lambda x: x.text, self.verses))


class Verse:
    
        def __init__(self, book_name: str, chapter_number: int, verse: Element) -> None:
            self.verse = verse
            self.book_name = book_name
            self.chapter_number = chapter_number
            self.number = int(verse.attrib["vnumber"])
            self.text = verse.text
    
        def __str__(self) -> str:
            return f"{self.book_name} {self.chapter_number}:{self.number}"