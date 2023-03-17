import os

from bs4 import BeautifulSoup
from bs4.element import Tag
from glob import glob
from datetime import datetime
from tqdm import tqdm
import chardet

from data_classes import NewsArticle
from db import Collections

if __name__ == "__main__":

    collections = Collections()

    folder_path = "data/reuters21578"
    paths = glob(os.path.join(folder_path, "*.sgm") )


    for path in paths:

        with open(path, "rb") as f:
            result = chardet.detect(f.read())
            
        with open(path, 'r', encoding=result["encoding"]) as f:
            soup = BeautifulSoup(f, "html.parser")

        articles: list[Tag] = soup.find_all("reuters")

        for article in tqdm(articles):
            

            try:
                title = article.find('title')
                if title is None:
                    continue
                else:
                    title = title.text.strip()
                content = article.find('body')
                if content is None:
                    continue
                else:
                    content = content.text.strip()
                if content == "":
                    continue
                date = article.find('date')
                if date is None:
                    continue
                else:
                    date = date.text.strip()
                date =  datetime.strptime(date, "%d-%b-%Y %H:%M:%S.%f").date()
                article = NewsArticle(title=title, content=content, date=date)
                article_json = article.to_json()
                collections.article_collection.insert_one(article_json)
            except Exception as e:
                continue