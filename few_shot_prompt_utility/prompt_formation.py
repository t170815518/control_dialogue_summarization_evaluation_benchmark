import random
import numpy as np
import nltk
import string
import spacy
from nltk.corpus import stopwords
import logging

nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")


def formulate_record_to_prompt_text(dialogue: str, model: str, summary: str = None, keyword_prompts: list = None,
                                    control_length: int = None, is_replace_entity: bool = False,
                                    is_add_instruction: bool = True, is_focus_planning: bool = False,
                                    is_flipped_focus: bool = False):
    """
    Formulate the dialogue and summary (optional) as the prompt text for model inference.
    :param keyword_prompts: list of keywords for entity control
    :param dialogue: str, the dialogue text
    :param summary: str, the summary text (optional)
    :return: str, the prompt text
    """
    speaker2replace_str = {}
    if is_add_instruction:
        if keyword_prompts is None:
            if control_length is None:
                prompt_text = 'Summarize the conversation:\n'
            else:
                prompt_text = 'Summarize the conversation with the defined length:\n'
        else:
            if is_focus_planning:
                prompt_text = 'Summarize the conversation with the focus perspectives provided:\n'
            else:
                prompt_text = 'Summarize the conversation with keywords:\n'
    else:
        prompt_text = ''
    dialogue = dialogue.strip().replace("\r", "")
    if is_replace_entity:  # if replace entity like DialogSum as #Person1#
        # Split each line with ':' (max_split=1), and get the set of speakers
        speakers = list(set([line.split(':', maxsplit=1)[0] for line in dialogue.split('\n')]))
        # keep non-empty element only
        speakers = [speaker for speaker in speakers if speaker != '']
        # create a dict for speaker2replace_str, e.g., {'Mary': '#Person1#'}
        speaker2replace_str = {speaker: f'#Person{speaker_id}#' for speaker_id, speaker in enumerate(speakers)}
        # replace the speaker name (e.g., 'mary', 'Mary') with the replace_str in the dialogue
        for speaker, replace_str in speaker2replace_str.items():
            dialogue = dialogue.replace(speaker, replace_str)
            dialogue = dialogue.replace(speaker.lower(), replace_str)
            try:
                dialogue = dialogue.replace(speaker[0].lower() + speaker[1:], replace_str)
            except IndexError:
                logging.info(f'IndexError: {speaker}')
    prompt_text += dialogue + '\n'
    if keyword_prompts is None:
        if control_length is None:
            prompt_text += 'Summary: '
        else:
            prompt_text += 'Summary with the length of {} words: '.format(control_length)
    else:
        if is_focus_planning:
            prompt_text += 'Summary with the focus perspectives {}: '.format(keyword_prompts)
        else:
            prompt_text += 'Summary with keywords {}: '.format(keyword_prompts)
    if summary:
        summary = summary.strip().replace("\n", "").replace("\r", "")
        if is_replace_entity:
            for speaker, replace_str in speaker2replace_str.items():
                summary = summary.replace(speaker, replace_str)
                summary = summary.replace(speaker.lower(), replace_str)
                summary = summary.replace(speaker[0].lower() + speaker[1:], replace_str)
        if is_flipped_focus:
            # replace the focus entity in summary with a random different name in keyword_prompts,
            # without using str.replace()
            summary = summary.split()
            for i, word in enumerate(summary):
                if word in keyword_prompts:
                    # the candidates exclude the name itself
                    candidates = [name for name in keyword_prompts if name != word]
                    if len(candidates) > 0:
                        summary[i] = random.choice(candidates)
                    else:
                        # log the candidates are empty
                        logging.info(f'candidates are empty: {keyword_prompts}')
                        continue
            summary = ' '.join(summary)
        prompt_text += summary
        if 'mt5' in model:
            prompt_text += summary + '</s>'
        return prompt_text
    else:
        if is_replace_entity:
            return prompt_text, speaker2replace_str
        else:
            return prompt_text


def format_prompt_from_demo_pairs(run_id2demo_pairs: dict, model: str, is_replace_entity: bool = False,
                                  is_add_instruction: bool = False, is_focus_planning: bool = False,
                                  is_random_label: bool = False, is_numerical_label: bool = False,
                                  is_flipped_label: bool = False, is_add_control_signals_in_demon: bool = True, ):
    """
    Format the prompt text from the demonstration pairs and save the prompt text to the file.
    Double newline is used to separate the prompt text for each dialogue.
    :param run_id2demo_pairs: dict, loaded from pre-generated pickle file
    :param model: str, the name of the model
    :return: run_id2prompts, run_id2gold_summaries
    """
    run_id2prompts = {}
    run_id2gold_summaries = {}
    try:
        summaries = [d_[1]['summary'] for d in run_id2demo_pairs.values() for d_ in d.values()]
        # convert summaries, a nested list, to a list
        summaries = [s for s_ in summaries for s in s_]
    except KeyError:
        summaries = None
    for run_id, demo_pairs in run_id2demo_pairs.items():
        run_id2prompts[run_id] = []
        run_id2gold_summaries[run_id] = []
        for test_id, elements in demo_pairs.items():
            if len(elements) == 2:  # when no keywords
                test_sample, demonstrations = elements
                assert is_numerical_label is False
                if is_add_instruction:
                    prompt = 'In this task, you are given an conversation. Your task is to summarize the ' \
                             'conversation.:\n'
                else:
                    prompt = ''
                gold_summary = test_sample['summary']
                test_id = 0
                if len(demonstrations) > 0:  # format the demonstration
                    for demo_id, (demo_dialogue, demo_summary) in enumerate(
                            zip(demonstrations['dialogue'], demonstrations['summary'])):
                        if is_add_instruction:
                            prompt += 'Example {}:\n'.format(demo_id + 1)
                        if is_random_label:
                            # randomly select a label from summaries as demo_summary
                            demo_summary = random.choice(summaries)
                        prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                  is_replace_entity=is_replace_entity,
                                                                  is_add_instruction=not is_add_instruction) + '\n' +\
                                  '\n'
                        test_id = demo_id
                test_id += 1
                if is_replace_entity:
                    prompt_text_, speaker2replace_str = formulate_record_to_prompt_text(test_sample['dialogue'], model,
                                                                                        is_replace_entity=is_replace_entity)
                    prompt += prompt_text_
                    # replace gold_summary with speaker2replace_str
                    for speaker, replace_str in speaker2replace_str.items():
                        gold_summary = gold_summary.replace(speaker, replace_str)
                        gold_summary = gold_summary.replace(speaker.lower(), replace_str)
                        gold_summary = gold_summary.replace(speaker[0].lower() + speaker[1:], replace_str)
                else:
                    if is_add_instruction:
                        prompt += 'Example {}:\n'.format(test_id + 1)
                    prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model,
                                                              is_add_instruction=not is_add_instruction)

                if 'mt5' in model:
                    prompt += '<extra_id_0>'
                run_id2prompts[run_id].append(prompt)

                run_id2gold_summaries[run_id].append(gold_summary)
            elif len(elements) == 3:  # when there are control signals
                assert is_replace_entity is False
                # check if 3rd element is list or int
                if isinstance(elements[2], list):  # keywords signals are provided
                    if is_add_instruction:
                        prompt = 'In this task, you are given an conversation. Your task is to summarize the ' \
                                 'conversation and include the keywords provided.:\n'
                    else:
                        prompt = ''
                    test_sample, demonstrations, keywords = elements
                    assert len(keywords) > 0
                    keywords_id = 0
                    if len(demonstrations) != 0:
                        # format the demonstration
                        for demo_dialogue, demo_summary in zip(demonstrations['dialogue'], demonstrations['summary']):
                            keywords_id += 1
                            if 'mt5' in model:
                                # double \n
                                # for space between demonstrations
                                if is_focus_planning:
                                    prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                              is_add_instruction=not is_add_instruction,
                                                                              is_flipped_focus=is_flipped_label) \
                                              + '\n' \
                                              + '\n'
                                else:
                                    prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                              is_focus_planning=is_focus_planning) + '\n' + '\n'
                            else:
                                if is_add_control_signals_in_demon:
                                    if is_focus_planning:
                                        prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                                  keyword_prompts=keywords[keywords_id],
                                                                                  is_add_instruction= not is_add_instruction,
                                                                                  is_focus_planning=is_focus_planning,
                                                                                  is_flipped_focus=is_flipped_label) \
                                                  + '\n' \
                                                  + '\n'
                                    elif is_numerical_label:
                                        # extract numerical information from demo_summary with spacy
                                        demo_keywords = []
                                        doc = nlp(demo_summary)
                                        for ent in doc.ents:
                                            if ent.label_ in ["TIME", "DATE", "QUANTITY", "PERCENT"]:
                                                demo_keywords.append(ent.text)  # fixme: may not be numerical
                                        if len(demo_keywords) == 0:
                                            # extract the numbers from summary
                                            demo_keywords = [word for word in nltk.word_tokenize(demo_summary)
                                                             if word.isdigit()]
                                        assert len(demo_keywords) > 0, 'demo summary does not cont' \
                                                                       'ain numerical information: {}'.format(demo_summary)
                                        prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                                  demo_keywords,
                                                                                  is_add_instruction= not is_add_instruction,
                                                                                  is_focus_planning=is_focus_planning) \
                                                  + '\n' \
                                                  + '\n'
                                    else:
                                        # use ntkl to extract not-stop-word keywords from demo_summary
                                        # and use them as the keywords for the prompt as the order of the keywords in the summary
                                        demo_keywords = nltk.word_tokenize(demo_summary)
                                        demo_keywords = [word for word in demo_keywords if
                                                         word.isalpha() and word not in STOP_WORDS]
                                        keyword_num = len(keywords[0])
                                        demo_keywords_id = np.random.choice(range(len(demo_keywords)), min(keyword_num,
                                                                                                           len(demo_keywords)),
                                                                            replace=False)
                                        demo_keywords_id = sorted(demo_keywords_id)
                                        demo_keywords = [demo_keywords[i] for i in demo_keywords_id]
                                        prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                                  demo_keywords,
                                                                                  is_add_instruction= not is_add_instruction,
                                                                                  is_focus_planning=is_focus_planning) \
                                                  + '\n' \
                                                  + '\n'
                                else:
                                    prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                              is_add_instruction=not is_add_instruction,
                                                                              is_focus_planning=is_focus_planning) \
                                              + '\n' \
                                              + '\n'
                    if 'mt5' in model:
                        prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model)
                        # join each keyword with strings '<extra_id_i>', where i is incrementing from 0
                        span_to_fill = '<extra_id_0> '  # empty space is needed
                        mask_id = 1
                        for keyword in keywords[0]:
                            span_to_fill += keyword + ' <extra_id_{}> '.format(mask_id)
                            mask_id += 1
                        prompt += span_to_fill.strip()
                        run_id2prompts[run_id].append([prompt, span_to_fill.strip()])
                    else:
                        # todo: keywords + name replaced?
                        prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model,
                                                                  keyword_prompts=keywords[0],
                                                                  is_focus_planning=is_focus_planning)
                        run_id2prompts[run_id].append([prompt, keywords[0]])
                    run_id2gold_summaries[run_id].append(test_sample['summary'])
                elif isinstance(elements[2], int):
                    if is_add_instruction:
                        prompt = 'In this task, you are given an conversation. Your task is to summarize the ' \
                                 'conversation into the text with pre-defined length.:\n'
                    else:
                        prompt = ''
                    test_sample, demonstrations, control_length = elements
                    for demo_dialogue, demo_summary in zip(demonstrations['dialogue'], demonstrations['summary']):
                        # count the words length (excluding punctuation) using ntlk
                        words = [word.lower() for word in nltk.word_tokenize(demo_summary) if word not in
                                 string.punctuation]
                        prompt += formulate_record_to_prompt_text(demo_dialogue, model, demo_summary,
                                                                  control_length=len(words)) \
                                  + '\n' \
                                  + '\n'
                    if 'mt5' in model:
                        prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model,
                                                                  control_length=control_length)
                        # join each keyword with strings '<extra_id_i>', where i is incrementing from 0
                        span_to_fill = '<extra_id_0> '  # empty space is needed
                        prompt += span_to_fill.strip()
                        run_id2prompts[run_id].append([prompt, span_to_fill.strip()])
                    else:
                        prompt += formulate_record_to_prompt_text(test_sample['dialogue'], model,
                                                                  control_length=control_length)
                        run_id2prompts[run_id].append([prompt, control_length])
                    run_id2gold_summaries[run_id].append(test_sample['summary'])
                else:
                    raise ValueError(
                            'The 3rd element as control signals of the demo pair is neither a list nor an int.')
            else:
                raise ValueError('The number of elements in the demo_pairs is not correct.')
    return run_id2prompts, run_id2gold_summaries
