import ast

import openai
import utils
import asyncio
from data_classes import Summary

import tiktoken


def calculate_token_size(tokenizer, text: str) -> int:
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_embedding(text: str):
    res = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    openai.ChatCompletion
    data = res["data"][0]["embedding"]
    return data


async def summarize_news_article(content: str, n_summaries: int) -> list[str]:
    prompts = [
        {
            "role": "system",
            "content": 
            f"""
you are a summarization bot.
You need to summarize the given content of the news article in one sentence.
Don't go beyond one sentence. 
You need to return {n_summaries} different versions of summary in list the list like this.
["result 1", "result 2", "result 3", "result 4", ...]
news article content will be in between <CONTENT><CONTENT> tags.
         """,
        },
        {"role": "user", "content": f"<CONTENT>{content}<CONTENT>"},
    ]
    res = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=prompts,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    finish_reason = res["choices"][0]["finish_reason"]

    if finish_reason == "length":
        summaries = res["choices"][0]["message"]["content"]
        prompts = prompts + [
            {"role": "assistant", "content": summaries},
            {"role": "user", "content": "go on please"},
        ]
        res = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=prompts,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        summaries = summaries + res["choices"][0]["message"]["content"]
    else:
        summaries = res["choices"][0]["message"]["content"]

    return ast.literal_eval(summaries)


async def summarize_bible(
    book_name: str, chapter_number: int, difficulty: str = "normal", max_lines: int = 10
):
    if difficulty == "easy":
        base_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"summarize Genesis chapter 1 under {max_lines} lines in easy terms.",
            },
            {
                "role": "assistant",
                "content": "God created the heavens and the earth. He spoke into existence light, the sky, the sea, land, vegetation, the sun, moon, and stars, sea creatures, birds, and land animals. Finally, God created humans in His own image, blessing them to be fruitful, multiply, and rule over the earth.",
            },
        ]
        propmt = base_messages + [
            {
                "role": "user",
                "content": f"summarize {book_name} chapter {chapter_number} under {max_lines} lines in easy terms",
            }
        ]
    elif difficulty == "normal":
        base_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"summarize Genesis chapter 1 under {max_lines} lines.",
            },
            {
                "role": "assistant",
                "content": "God created the heavens and the earth. He spoke into existence light, the sky, the sea, land, vegetation, the sun, moon, and stars, sea creatures, birds, and land animals. Finally, God created humans in His own image, blessing them to be fruitful, multiply, and rule over the earth.",
            },
            {
                "role": "user",
                "content": f"summarize Genesis chapter 2 under {max_lines} lines",
            },
            {
                "role": "assistant",
                "content": "God finished creating the heavens and earth, and on the seventh day, He rested. He formed Adam out of dust and placed him in the Garden of Eden, giving him the task of tending to it. God also created animals and birds, but no suitable partner for Adam. So, God made Eve from Adam's rib to be his helper and companion. They were both naked and unashamed.",
            },
        ]
        propmt = base_messages + [
            {
                "role": "user",
                "content": f"summarize {book_name} chapter {chapter_number} under {max_lines} lines",
            }
        ]
    else:
        raise ValueError("difficulty must be either 'easy' or 'normal'")

    res = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=propmt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return Summary(
        difficulty=difficulty,
        max_lines=max_lines,
        summary=res["choices"][0]["message"]["content"],
    )
