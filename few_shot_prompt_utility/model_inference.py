import re
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prompt_T5(mt5_model, mt5_tokenizer, prompt_text: str) -> str:
    """
    Makes inference using T5 (or mT5) model given the prompt text
    :param mt5_model:
    :param mt5_tokenizer:
    :param prompt_text: str
    :return: the response text
    """
    # tokenize dialogue data
    X = mt5_tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    # Summarize
    y_ids = mt5_model.generate(X, max_length=50, do_sample=False, eos_token_id=2, early_stopping=True, num_beams=5)
    y = mt5_tokenizer.decode(y_ids[0], skip_special_tokens=True)
    # remove r"<extra_id_\d+>" from y
    y = re.sub(r"<extra_id_\d+>", "", y)
    return y
