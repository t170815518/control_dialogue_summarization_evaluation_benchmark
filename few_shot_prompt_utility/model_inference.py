import logging
import re
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prompt_llm(model, tokenizer, prompt_text: str, is_gpt_style: bool = False, spans_to_fill = None) -> str:
    """
    Makes inference using GPT-style or T5-style llm given the prompt text
    :param model:
    :param tokenizer:
    :param prompt_text: str
    :param is_gpt_style: bool, True means the model is GPT-style, False means a T5-style llm
    :return: the response text
    """
    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0]
    # tokenize dialogue data
    device_count = torch.cuda.device_count()
    if device_count > 1:
        X = tokenizer.encode(prompt_text, return_tensors="pt").to(device_count-1)
    else:
        X = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    # Summarize
    if is_gpt_style:
        y_ids = model.generate(X, num_beams=5,
                               max_new_tokens=50, early_stopping=True,
                               no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)  # ref:
        # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to
        # -eos-token-id
    else:
        y_ids = model.generate(X, max_length=50, do_sample=False, eos_token_id=2, early_stopping=True, num_beams=5)
    y = tokenizer.decode(y_ids[0], skip_special_tokens=True)
    raw_output = y
    # parse the generated text
    if is_gpt_style:
        y = re.split(r'Summary(( with keywords \[.+\])|( with the length of \d+ words))?:', y)[-1].strip()
        return y
    else:  # remove r"<extra_id_\d+>" from y
        # if kwargs has the key 'spans_to_fill'
        logging.info('generated content {}'.format(y))
        if spans_to_fill is None:  # direct prompt of mT5
            y = re.sub(r"<extra_id_\d+>", "", y)
        else:
            # split the response by r'<extra_id_\d+>'
            spans_filled = re.split(r'<extra_id_\d+>', y)
            spans_filled = [x.strip() for x in spans_filled[1:]]
            # replace span_to_fill with spans_filled
            formatted_response = spans_to_fill
            for i in range(len(spans_filled)):
                formatted_response = formatted_response.replace('<extra_id_{}>'.format(i), spans_filled[i], 1)
            y = formatted_response
        return y, raw_output

