import logging
import numpy as np
from rouge_score import rouge_scorer


def evaluate_response_summaries(run_id2pred_summaries: dict, run_id2gold_summaries: dict, run_id2prompts: dict):
    """
    Evaluate the quality of the generated summaries.
    :param run_id2pred_summaries: a dictionary of run_id to a list of generated summaries
    :param run_id2gold_summaries: a dictionary of run_id to a list of gold summaries
    :param run_id2prompts: a dictionary of run_id to a list of prompts
    :return: summary_table, summary_text_table, both are pd.DataFrame
    """
    # evaluate the model
    summary_table = []
    summary_text_table = []
    for run_id, pred_summaries in run_id2pred_summaries.items():
        scores_in_run = []
        logging.info("Evaluating the model for Run {}".format(run_id))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for i, pred_summary in enumerate(pred_summaries):
            gold_summary = run_id2gold_summaries[run_id][i]
            scores = scorer.score(pred_summary, gold_summary)
            scores_in_run.append(scores)
            summary_text_row = {
                    'run_id': run_id,
                    'prompt': run_id2prompts[run_id][i],
                    'pred_summary': pred_summary,
                    'gold_summary': gold_summary
                    }
            summary_text_table.append(summary_text_row)
        summary_row = {
                'run_id': run_id,
                'rouge_1_precision': np.mean([score['rouge1'].precision for score in scores_in_run]),
                'rouge_1_recall': np.mean([score['rouge1'].recall for score in scores_in_run]),
                'rouge_1_fmeasure': np.mean([score['rouge1'].fmeasure for score in scores_in_run]),
                'rouge_2_precision': np.mean([score['rouge2'].precision for score in scores_in_run]),
                'rouge_2_recall': np.mean([score['rouge2'].recall for score in scores_in_run]),
                'rouge_2_fmeasure': np.mean([score['rouge2'].fmeasure for score in scores_in_run]),
                'rouge_L_precision': np.mean([score['rougeL'].precision for score in scores_in_run]),
                'rouge_L_recall': np.mean([score['rougeL'].recall for score in scores_in_run]),
                'rouge_L_fmeasure': np.mean([score['rougeL'].fmeasure for score in scores_in_run]),
                }
        summary_table.append(summary_row)

    return summary_table, summary_text_table
