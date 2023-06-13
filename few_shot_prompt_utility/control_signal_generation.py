import logging
import nltk
import string
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset



nltk.download('stopwords')
nltk.download('punkt')


def generate_tf_idf_keywords(run_id2demo_pairs, k: int):
    """
    :param run_id2demo_pairs:
    :param k: the number of keywords, i.e., top k tf-idf words
    :return: run_id2demo_pairs with new key 'keywords'
    fixme: check the keywords are in the order
    """
    assert k > 0
    logging.info('Extracting top {} tf-idf keywords from the training dataset'.format(k))

    # Initialize a TfidfVectorizer with the desired configuration
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words='english')

    # get the OrderDict, key=test_id, value=summary content
    test_summaries_in_order = [(test_id, demonstrations[0]['summary']) for test_id, demonstrations in
                               run_id2demo_pairs[0].items()]
    summaries = [x[1] for x in test_summaries_in_order]
    # Fit the vectorizer on the text data
    vectorizer.fit(summaries)
    # Extract the tf-idf matrix
    tfidf_matrix = vectorizer.transform(summaries)
    # Extract the feature names (i.e., the words) from the vectorizer
    try:
        feature_names = vectorizer.get_feature_names()
    except AttributeError:  # newer sklearn uses another function name,
        # ref: https://stackoverflow.com/questions/70215049/attributeerror-tfidfvectorizer-object-has-no-attribute
        # -get-feature-names-out
        feature_names = vectorizer.get_feature_names_out()
    # For each data record, extract the top k keywords based on their tf-idf scores
    test_id2keywords = {}
    for x in range(len(summaries)):
        tfidf_scores = tfidf_matrix[x].toarray()[0]
        top_indices = tfidf_scores.argsort()[-k:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        # sort top_keywords as the order of their appearance in the summary
        top_keywords = sorted(top_keywords, key=lambda keyword: summaries[x].lower().find(keyword))
        test_id2keywords[test_summaries_in_order[x][0]] = top_keywords

    for run_id, demo_pairs in run_id2demo_pairs.items():
        for test_id, _ in demo_pairs.items():
            element = list(run_id2demo_pairs[run_id][test_id])
            element.append([test_id2keywords[test_id]])
            run_id2demo_pairs[run_id][test_id] = element

    return run_id2demo_pairs


def generate_control_length(run_id2demo_pairs):
    logging.info('Extracting length as control signals')

    for run_id, demo_pairs in run_id2demo_pairs.items():
        for test_id, _ in demo_pairs.items():
            element = list(run_id2demo_pairs[run_id][test_id])
            # get the word length of the gold summary
            gold_summary = element[0]['summary']
            # count the words length (excluding punctuation) using ntlk
            words = [word.lower() for word in nltk.word_tokenize(gold_summary) if word not in
                     string.punctuation]
            control_length = len(words)
            element.append(control_length)
            run_id2demo_pairs[run_id][test_id] = element

    return run_id2demo_pairs


def generate_focus_planning(run_id2demo_pairs):
    """
    :return: run_id2demo_pairs with new key 'keywords'
    """
    logging.info('Extracting focus planning as control signals')
    train_target2focus = {}
    test_target2focus = {}
    # read the control signal from SAMsum_data_OCC/test.source and SAMsum_data_OCC/train.source
    with open('SAMsum_data_OCC/test.source', 'r', encoding='utf-8') as f, \
            open('SAMsum_data_OCC/test.target', 'r') as target_f:
        test_source = f.readlines()
        test_target = target_f.readlines()
        assert len(test_source) == len(test_target)
        for line, target in zip(test_source, test_target):
            target = target.strip()
            # extract the content within the curly bracket
            focus_str = line[line.find('{') + 1:line.find('}')]
            # split by '|'
            focus_list = focus_str.split('|')
            # strip the leading and trailing spaces
            focus_list = [x.strip() for x in focus_list]
            test_target2focus[target] = focus_list

    with open('SAMsum_data_OCC/train.source', 'r', encoding='utf-8') as f, \
            open('SAMsum_data_OCC/train.target','r', encoding='utf-8') as target_f:
        train_source = f.readlines()
        train_target = target_f.readlines()
        assert len(train_source) == len(train_target)
        for line, target in zip(train_source, train_target):
            target = target.strip()
            # extract the content within the curly bracket
            focus_str = line[line.find('{') + 1:line.find('}')]
            # split by '|'
            focus_list = focus_str.split('|')
            # strip the leading and trailing spaces
            focus_list = [x.strip() for x in focus_list]
            train_target2focus[target] = focus_list

    error_count = 0  # count the number of test summaries that are not in the training set
    # deep copy run_id2demo_pairs
    run_id2demo_pairs_copy = copy.deepcopy(run_id2demo_pairs)
    for run_id, demo_pairs in run_id2demo_pairs_copy.items():
        for test_id, (test_dialogue, demonstrations) in demo_pairs.items():
            try:
                element = list(run_id2demo_pairs_copy[run_id][test_id])
                focus_signals = [test_target2focus[test_dialogue['summary'].strip()]] + \
                                [train_target2focus[train_summary.strip()] for train_summary in demonstrations['summary']]
                element.append(focus_signals)
                run_id2demo_pairs[run_id][test_id] = element
            except KeyError:
                logging.warning(f'KeyError: {test_dialogue["summary"].strip()}')
                # remove run_id2demo_pairs[run_id][test_id] from run_id2demo_pairs
                del run_id2demo_pairs[run_id][test_id]
                continue

    return run_id2demo_pairs


if __name__ == '__main__':
    generate_focus_planning({})
