from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET


def get_book_name_by_index(bible: Element, index: int) -> str:
    return bible[index].attrib["bname"]

def get_book_index_by_name(bible: Element, name: str) -> int:
    books: list[Element] = bible.findall("BIBLEBOOK")
    for i, book in enumerate(books):
        if book.attrib["bname"].lower() == name.lower():
            return i
    return -1

def get_verse_text(bible: Element, book: any, chapter: int, verse: int) -> str:
    if type(book) == str:
        book: Element = bible[get_book_index_by_name(bible, book)]
    elif type(book) == int:
        book: Element = bible[book]
    else:
        raise TypeError("book must be either a string or an integer")

    chapter: Element = book.findall("CHAPTER")[chapter - 1]
    verse: Element = chapter.findall("VERS")[verse - 1]
    return verse.text

def get_chapter_text(bible: Element, book: any, chapter: int) -> list[str]:
    if type(book) == str:
        book: Element = bible[get_book_index_by_name(bible, book)]
    elif type(book) == int:
        book: Element = bible[book]
    else:
        raise TypeError("book must be either a string or an integer")

    chapter: Element = book.findall("CHAPTER")[chapter - 1]
    verses: list[Element] = chapter.findall("VERS")
    verse_texts: list[str] = []

    for verse in verses:
        verse_texts.append(verse.text)

    return verse_texts

def get_book_text(bible: Element, book: any) -> list[list[str]]:
    if type(book) == str:
        book: Element = bible[get_book_index_by_name(bible, book)]
    elif type(book) == int:
        book: Element = bible[book]
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






if __name__ == "__main__":
    bible: Element = ET.parse("data/nrsv_bible.xml").getroot()
    books: list[Element] = bible.findall("BIBLEBOOK")

    print(get_book_name_by_index(bible, 0))
    print(get_book_index_by_name(bible, "Genesis"))
    print(get_verse_text(bible, "Genesis", 1, 1))
    print(get_chapter_text(bible, "Genesis", 1))
    print(get_book_text(bible, "Genesis"))