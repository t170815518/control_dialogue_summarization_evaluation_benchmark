import logging
import string
from sklearn.feature_extraction.text import TfidfVectorizer


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
