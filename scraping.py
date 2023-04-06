import requests
from bs4 import BeautifulSoup
from pprint import pprint
import json
from tqdm import tqdm

def scrape_and_save():
    url = "https://biblesummary.info"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    links = soup.select("#page_body > div.column_side.page_content > ul > li > a")
    summary_urls = [url + link.get('href') for link in links]
    all_summaries = []

    for summary_url in tqdm(summary_urls):
        summary_page = requests.get(summary_url)
        summary_soup = BeautifulSoup(summary_page.content, "html.parser")
        summaries = summary_soup.select("#page_body > div.column_main.page_content > div > p.tweet_content")
        summary_texts = [summary.text for summary in summaries]
        summary_texts = ["".join(text.split(":")[1:]).strip()  for text in summary_texts]
        all_summaries.extend(summary_texts)

    with open("bible_summaries.json", "w") as f:
        json.dump(all_summaries, f)


if __name__ == "__main__":
    with open("bible_summaries.json", "r") as f:
        summaries = json.load(f)


    print(len(summaries))
    