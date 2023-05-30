import random
import numpy as np
import nltk
from nltk.corpus import stopwords
import logging


nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = set(stopwords.words('english'))


def formulate_record_to_prompt_text(dialogue: str, model: str, summary: str = None, keyword_prompts: list = None):
    """
    Formulate the dialogue and summary (optional) as the prompt text for model inference.
    :param dialogue: str, the dialogue text
    :param summary: str, the summary text (optional)
    :return: str, the prompt text
    """
    if keyword_prompts is None:
        prompt_text = 'Summarize the conversation:\n'
    else:
        prompt_text = 'Summarize the conversation with keywords:\n'
    dialogue = dialogue.strip().replace("\r", "")
    prompt_text += dialogue + '\n'
    if keyword_prompts is None:
        prompt_text += 'Summary: '
    else:
        prompt_text += 'Summary with keywords {}: '.format(keyword_prompts)
    if summary:
        summary = summary.strip().replace("\n", "").replace("\r", "")
        prompt_text += summary
        if 'mt5' in model:
            prompt_text += summary + '</s>'
    return prompt_text


def format_prompt_from_demo_pairs(run_id2demo_pairs: dict, model: str):
    """
    Format the prompt text from the demonstration pairs and save the prompt text to the file.
    :param run_id2demo_pairs: dict, loaded from pre-generated pickle file
    :param model: str, the name of the model
    :return: run_id2prompts, run_id2gold_summaries
    """
    run_id2prompts = {}
    run_id2gold_summaries = {}
    for run_id, demo_pairs in run_id2demo_pairs.items():
        run_id2prompts[run_id] = []
        run_id2gold_summaries[run_id] = []
        for test_id, elements in demo_pairs.items():
            if len(elements) == 2:  # when no keywords
                test_sample, demonstrations = elements
                prompt = ''
                for demo_dialogue, demo_summary in zip(demonstrations['dialogue'], demonstrations['summary']):
                    prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary) + '\n' + '\n'  # double \n
                    # for space between demonstrations
                prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model)
                if 'mt5' in model:
                    prompt += '<extra_id_0>'
                run_id2prompts[run_id].append(prompt)
                run_id2gold_summaries[run_id].append(test_sample['summary'])
            elif len(elements) == 3:
                test_sample, demonstrations, keywords = elements
                prompt = ''
                for demo_dialogue, demo_summary in zip(demonstrations['dialogue'], demonstrations['summary']):
                    if 'mt5' in model:
                        # double \n
                        # for space between demonstrations
                        prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary) + '\n' + '\n'
                    else:
                        # use ntkl to extract not-stop-word keywords from demo_summary
                        # and use them as the keywords for the prompt as the order of the keywords in the summary
                        demo_keywords = nltk.word_tokenize(demo_summary)
                        demo_keywords = [word for word in demo_keywords if word.isalpha() and word not in STOP_WORDS]
                        keyword_num = len(keywords[0])
                        demo_keywords_id = np.random.choice(range(len(demo_keywords)), min(keyword_num,
                                                                                           len(demo_keywords)),
                                                            replace=False)
                        demo_keywords_id = sorted(demo_keywords_id)
                        demo_keywords = [demo_keywords[i] for i in demo_keywords_id]
                        prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary, demo_keywords) \
                                  + '\n' \
                                  + '\n'
                if 'mt5' in model:
                    prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model)
                    # join each keyword with strings '<extra_id_i>', where i is incrementing from 0
                    span_to_fill = '<extra_id_0> '  # empty space is needed
                    mask_id = 1
                    if len(keywords) > 1:  # unexpected behavior
                        logging.warning(f'Number of keywords is {len(keywords)} for run_id {run_id} test_id {test_id}')
                    for keyword in keywords[0]:
                        span_to_fill += keyword + ' <extra_id_{}> '.format(mask_id)
                        mask_id += 1
                    prompt += span_to_fill.strip()
                    run_id2prompts[run_id].append([prompt, span_to_fill.strip()])
                else:
                    prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model,
                                                              keyword_prompts=keywords[0])
                    run_id2prompts[run_id].append([prompt, keywords[0]])
                run_id2gold_summaries[run_id].append(test_sample['summary'])
            else:
                raise ValueError('The number of elements in the demo_pairs is not correct.')
    return run_id2prompts, run_id2gold_summaries
