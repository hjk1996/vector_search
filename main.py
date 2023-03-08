import openai

import openai_functions
import db
import asyncio
from bible import Bible
import utils
from tqdm import tqdm


async def summarize_and_save(difficulty: str, max_lines: int) -> None:
    openai.api_key = utils.load_api_key()
    collection = db.BibleSummaryCollection()
    bible = Bible("data/nrsv_bible.xml")

    for book in bible.books:
        for chapter in tqdm(book, desc=f"Summarizing {book.name}"):
            summary = await openai_functions.summarize_bible(
                book_name=book.name,
                chapter_number=chapter.number,
                difficulty=difficulty,
                max_lines=max_lines,
            )
            collection.collection.update_one(
                {"book": book.name, "chapter": chapter.number},
                {"$push": {f"{max_lines}_line_summaries": summary.to_json()}},
            )


async def main():
    openai.api_key = utils.load_api_key()
    collection = db.BibleSummaryCollection()
    bible = Bible("data/nrsv_bible.xml")

    for book in bible.books:
        for chapter in tqdm(book, desc=f"Summarizing {book.name}"):
            summary = await openai_functions.summarize_bible(
                book_name=book.name,
                chapter_number=chapter.number,
                difficulty="easy",
                max_lines=1,
            )
            collection.collection.update_one(
                {"book": book.name, "chapter": chapter.number},
                {"$push": {"one_line_summaries": summary.to_json()}},
            )


asyncio.run(main())
