"""Machine learning class that separates the training logic from the main script.

Author: Cas van Rijbroek, Lex Bosch, Sophie Hospel
Date: 18-03-20
"""

import pickle

from sklearn.svm import SVC

from Objects.proteins import Protein
from calculate_scores import get_score_list
from sklearn.model_selection import train_test_split


class BigBrainMachineLearning:
    def __init__(self, data_file, title):
        """Constructor that trains a new classifier and pickles it.

        :param data_file: Name of the file containing the protein data.
        :param title: Job title.
        """
        self.title = title
        self.train_ids, self.test_ids, self.train_sequences, self.test_sequences = self.parse_data(data_file)
        self.classifier = self.train_module()
        self.dump_classifier()

    def train_module(self):
        """Uses the support vector machine functionality of sklearn to train a new classifier.

        :return: The new classifier.
        """
        weighted_sequences = get_score_list(self.train_sequences)
        classifier = SVC(kernel="rbf")
        classifier.fit(weighted_sequences, self.train_ids)

        return classifier

    def parse_data(self, data_file):
        """Uses internal methods to parse the protein data into train and test sets.

        :param data_file: Name of the file containing the protein data.
        :return: Tuple containing all the train and test ids and sequences.
        """
        content_file = []

        with open(data_file, "r") as input_file:
            opened_fasta_file = input_file.read()

        for single_prot_segment in opened_fasta_file.split(">"):
            content_file.append(single_prot_segment)

        proteins = self.raw_to_protein(content_file[1:])
        all_ids, all_sequences = self.get_ids_sequences(proteins)

        return train_test_split(all_ids, all_sequences, test_size=0.20, random_state=42)

    def raw_to_protein(self, raw_data):
        """Transforms the raw data into protein objects.

        :param raw_data: The raw data.
        :return: The protein objects.
        """
        proteins = []
        for single_segment in raw_data:
            head, seq, classifier = single_segment.split("\n")[:-1]
            proteins.append(Protein(head, seq, classifier))

        return proteins

    def get_ids_sequences(self, proteins):
        """Retrieves the class ids and sequences from protein objects.

        :param proteins: The protein objects.
        :return: Class ids and sequences belonging to the protein objects.
        """
        class_ids = []
        sequences = []

        for single_object in proteins:
            class_ids.append(single_object.type)
            sequences.append(single_object.sequence_list)

        for i in range(len(class_ids)):
            if class_ids[i] != "NO_SP":
                class_ids[i] = "SP"

        return class_ids, sequences

    def dump_classifier(self):
        """Pickles the classifier."""
        with open(f'Files/{self.title}.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
