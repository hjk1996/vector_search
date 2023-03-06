# calculate the number of tokens in a list of strings. it will be used for openai's api
def calculate_token_size(text: list[str]) -> int:
    return len(" ".join(text).replace(" ", "").strip())


def seperate_text_in_half(text: list[str]) -> list[list[str]]:
    half = len(text) // 2
    return [text[:half], text[half:]]


def load_api_key() -> str:
    with open("api_key.txt", "r") as f:
        return f.read().strip()