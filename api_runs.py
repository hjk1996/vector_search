import ast

import openai


import openai_functions
from openai_functions import calculate_token_size
import db
import asyncio
from bible import Bible
import utils
from tqdm import tqdm
import tiktoken


async def summarize_bible_and_save(difficulty: str, max_lines: int) -> None:
    openai.api_key = utils.load_api_key()
    collections = db.Collections()
    bible = Bible("data/nrsv_bible.xml")

    for book in bible.books:
        for chapter in tqdm(book, desc=f"Summarizing {book.book_name}"):
            summary = await openai_functions.summarize_bible(
                book_name=book.book_name,
                chapter_number=chapter.chapter_number,
                difficulty=difficulty,
                max_lines=max_lines,
            )
            collections.bible_collection.update_one(
                {"book": book.book_name, "chapter": chapter.chapter_number},
                {"$push": {f"{max_lines}_line_summaries": summary.to_json()}},
            )

def calculate_total_tokens():
    collections = db.Collections()
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    cursor = collections.article_collection.find()
    total_tokens = 0
    for doc in tqdm(cursor):
        content = doc["content"]
        token_size = calculate_token_size(tokenizer, content)
        total_tokens += token_size
    print(total_tokens)

async def summarize_articles_and_save():
    openai.api_key = utils.load_api_key()
    collections = db.Collections()

    cursor = collections.article_collection.find({"summaries": None})
    for doc in tqdm(cursor):
        try:
            content = doc["content"]
            summaries = await openai_functions.summarize_news_article(content, 4)
            collections.article_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"summaries": summaries}},
            )
        except Exception as e:
            continue


async def main():
    await summarize_articles_and_save()


asyncio.run(main())
