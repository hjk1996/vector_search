import openai

import openai_functions
import db
import asyncio
from bible import Bible
import utils
from tqdm import tqdm


async def main():
    openai.api_key = utils.load_api_key()
    collection = db.BibleSummaryCollection()
    bible = Bible("data/nrsv_bible.xml")

    for book in bible[22:]:
        for chapter in tqdm(book, desc=f"Processing {book.name} - {book.number}") :
            res = await openai_functions.summarize_bible(
                book_name=chapter.book_name, chapter_number=chapter.number, max_lines=10, difficulty="easy"
            )
            query = {
                "book_number": book.number,
                "chapter": chapter.number,
            }
            new_values = {
                "$push": {
                    "summaries": res.to_json()
                }
            }
            collection.collection.update_one(query, new_values)


asyncio.run(main())
