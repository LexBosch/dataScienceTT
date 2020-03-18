"""Script to add weights to protein sequences based on various sequence attributes.

Author: Cas van Rijbroek, Lex bosch, Sophie Hospel
Date: 18-03-20
"""
import json


def get_score_list(sequences):
    """Adds weights to sequences.
    
    :param sequences: List of protein sequences.
    :return: List of weighted protein sequences.
    """
    with open("scores.json", "r") as input_files:
        scores = json.load(input_files)
    
    arbitrair = scores["arbitrair"]
    arbitrair_polair = scores["arbitrair_polair"]

    all_scores_list = []
    segment_size = 10

    for single_sequence in sequences:
        score = []
        single_sequence = single_sequence[0:40]
        for i in range(0, (len(single_sequence) - segment_size)):
            sub_seq = single_sequence[i:i + segment_size]
            total_score = round(sum(arbitrair[amino] for amino in sub_seq), 2)
            score.append(total_score)
        all_scores_list.append(score)

    return all_scores_list
