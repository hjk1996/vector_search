import torch
import transformers


from bible import Bible
from embeddings import load_embeddings
from models import ModelWrapper


if __name__ == "__main__":
    bible = Bible("data/nrsv_bible.xml", "data/chapter_index_map.json")
    device = torch.device("cpu")
    embeddings = load_embeddings("embeddings", device)
    gpt2_xl = transformers.GPT2Model.from_pretrained("gpt2-xl").to("cpu")
    gpt2_xl_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2-xl")

    gpt2_xl_model = ModelWrapper(
        model=gpt2_xl,
        tokenizer=gpt2_xl_tokenizer,
        bible=bible,
        embedding=embeddings[model_name],
        name=model_name,
        device=device,


    )

    print(gpt2_xl_model.get_related_n_chapters("a man encourages the people to rebuild the temple and promises God's blessing", 5))
