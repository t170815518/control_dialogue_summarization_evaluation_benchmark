import re
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prompt_llm(model, tokenizer, prompt_text: str, is_gpt_style: bool = False) -> str:
    """
    Makes inference using GPT-style or T5-style llm given the prompt text
    :param model:
    :param tokenizer:
    :param prompt_text: str
    :param is_gpt_style: bool, True means the model is GPT-style, False means a T5-style llm
    :return: the response text
    """
    # tokenize dialogue data
    X = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    # Summarize
    if is_gpt_style:
        y_ids = model.generate(X, num_beams=5,
                               max_new_tokens=50, early_stopping=True,
                               no_repeat_ngram_size=2)
    else:
        y_ids = model.generate(X, max_length=50, do_sample=False, eos_token_id=2, early_stopping=True, num_beams=5)
    y = tokenizer.decode(y_ids[0], skip_special_tokens=True)
    if is_gpt_style:
        y = y.strip().split('Summary:')[-1].strip()
    else:  # remove r"<extra_id_\d+>" from y
        y = re.sub(r"<extra_id_\d+>", "", y)
    return y
