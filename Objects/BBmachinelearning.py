from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from time import time
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from math import fsum
import pickle

from plzwerk import get_ugly_list


class BigBrainMachineLearning:
    def __init__(self, class_ids: list, sequences: list):
        self.class_ids = class_ids
        self.count_vect = CountVectorizer()
        self.sequences = sequences
        self.classifier = self.train_module() ## wat is die self classifier?

    def train_module(self):
        """
        Vult vectoren met eigenschappen
        :return:
        """

        x_train_counts = self.count_vect.fit_transform(self.sequences)

        ja = get_ugly_list(self.sequences)

        #time()
        plzwerk = SVC(kernel="linear")
        plzwerk.fit(ja, self.class_ids)
        #time()

        with open('filename3.pickle', 'wb') as handle:
            pickle.dump(plzwerk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_classifier(self):
        return self.classifier

    def get_count_vect(self):
        return self.count_vect
