from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET


class Bible:
    def __init__(self, file_path: str) -> None:
        self.root: Element = ET.parse(file_path).getroot()
        self.books = self.root.findall("BIBLEBOOK")

    def get_book_name_by_index(self, index: int) -> str:
        return self.root[index].attrib["bname"]

    def get_book_index_by_name(self, name: str) -> int:
        for i, book in enumerate(self.books):
            if book.attrib["bname"].lower() == name.lower():
                return i
        return -1

    def get_verse_text(self, book: any, chapter: int, verse: int) -> str:
        if type(book) == str:
            book: Element = self.root[self.get_book_index_by_name(book)]
        elif type(book) == int:
            book: Element = self.root[book]
        else:
            raise TypeError("book must be either a string or an integer")

        chapter: Element = book.findall("CHAPTER")[chapter - 1]
        verse: Element = chapter.findall("VERS")[verse - 1]
        return verse.text

    def get_chapter_text(self, book: any, chapter: int) -> list[str]:
        if type(book) == str:
            book: Element = self.root[self.get_book_index_by_name(book)]
        elif type(book) == int:
            book: Element = self.root[book]
        else:
            raise TypeError("book must be either a string or an integer")

        chapter: Element = book.findall("CHAPTER")[chapter - 1]
        verses: list[Element] = chapter.findall("VERS")
        verse_texts: list[str] = []

        for verse in verses:
            verse_texts.append(verse.text)

        return verse_texts

    def get_book_text(self, book: any) -> list[list[str]]:
        if type(book) == str:
            book: Element = self.root[self.get_book_index_by_name(book)]
        elif type(book) == int:
            book: Element = self.root[book]
        else:
            raise TypeError("book must be either a string or an integer")

        chapters: list[Element] = book.findall("CHAPTER")
        chapter_texts: list[list[str]] = []

        for chapter in chapters:
            verses: list[Element] = chapter.findall("VERS")
            verse_texts: list[str] = []

            for verse in verses:
                verse_texts.append(verse.text)

            chapter_texts.append(verse_texts)

        return chapter_texts