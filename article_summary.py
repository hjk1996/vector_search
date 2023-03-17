from pprint import pprint

from tqdm import tqdm
from transformers import pipeline, PegasusForConditionalGeneration, AutoTokenizer

from db import Collections


if __name__ == "__main__":
    collections = Collections()

    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

    ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
    inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")

    summary_ids = model.generate(inputs["input_ids"], num_beams=10,  num_return_sequences=5, max_length=200)
    results = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pprint(results)

    # cursor = collections.article_collection.find()


    # for article in tqdm(cursor):
    #     content = article["content"]
    #     summary = summarizer(content, max_length=200, min_length=30, do_sample=True, num_beams=4, early_stopping=True, no_repeat_ngram_size=2, num_return_sequences=4)
    #     summaries = [s["summary_text"] for s in summary]
    #     pprint(summaries[2])
    #     break  
   