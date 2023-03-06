import openai
import utils


def summarize_text(text: str) -> str:
    text_list = [text]
    n_tokens = utils.calculate_token_size(text)
    if n_tokens > 10000:
        text_list = utils.seperate_text_in_half(text)

    res_texts = []
    for t in text_list:
        t = f"summarize this text: {t}"
        response = openai.Completion.create(
            engine="davinci",
            prompt=t,
            temperature=0.7,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        res_texts.append(response['choices'][0]['text'])

    return " ".join(res_texts)

def get_embedding(text: str):
    res =  openai.Embedding.create(input = [text], model="text-embedding-ada-002")
    openai.ChatCompletion
    data = res['data'][0]['embedding']
    return data