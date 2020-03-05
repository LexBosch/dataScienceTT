"""
Author: AI experts
Date: 05-03-20
"""


def get_ugly_list(sequences):
    arbitrair = {"R": -4.5,
                 "K": -3.9,
                 "N": -3.5,
                 "D": -3.5,
                 "Q": -3.5,
                 "E": -3.5,
                 "H": -3.2,
                 "P": -1.6,
                 "Y": -1.3,
                 "W": -0.9,
                 "S": -0.8,
                 "T": -0.7,
                 "G": -0.4,
                 "A": 1.8,
                 "M": 1.9,
                 "C": 2.5,
                 "F": 2.8,
                 "L": 3.8,
                 "V": 4.2,
                 "I": 4.5}

    arbitrair_polair = {
        "A": 6.00,
        "R": 10.76,
        "N": 5.41,
        "D": 2.77,
        "C": 5.07,
        "E": 3.22,
        "Q": 5.65,
        "G": 5.97,
        "H": 7.59,
        "I": 6.02,
        "L": 5.98,
        "K": 9.74,
        "M": 5.74,
        "F": 5.48,
        "P": 6.3,
        "U": 5.68,
        "S": 5.68,
        "T": 5.6,
        "W": 5.89,
        "Y": 5.66,
        "V": 5.96
    }

    ja = []
    groote_van_segment = 10
    for single_sequence in sequences:
        ja1 = []
        single_sequence = single_sequence[0:40]
        for i in range(0, (len(single_sequence) - groote_van_segment)):
            sub_seq = single_sequence[i:i + groote_van_segment]
            total_score = round(sum(arbitrair[amino] for amino in sub_seq), 2)
            ja1.append(total_score)
        ja.append(ja1)
    return ja
