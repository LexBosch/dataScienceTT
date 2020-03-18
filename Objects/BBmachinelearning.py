"""
Author: AI experts
Date: 05-03-20
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from calculate_scores import get_score_list
import pickle


class BigBrainMachineLearning:
    """
    Class to train the given data.
    """

    def __init__(self, class_ids: list, sequences: list):
        self.class_ids = class_ids
        self.count_vect = CountVectorizer()
        self.sequences = sequences
        self.classifier = self.train_module() ## wat is die self classifier? veranderen naar self.train_module()?

    def train_module(self):
        x_train_counts = self.count_vect.fit_transform(self.sequences)
        ja = get_score_list(self.sequences)
        print(ja)
        plzwerk = SVC(kernel="linear")
        plzwerk.fit(ja, self.class_ids)

        with open('filename3.pickle', 'wb') as handle:
            pickle.dump(plzwerk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_classifier(self):
        return self.classifier

    def get_count_vect(self):
        return self.count_vect
