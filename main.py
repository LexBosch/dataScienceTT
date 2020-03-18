"""Main script that controls the machine learning pipeline.

Author: Cas van Rijbroek, Lex Bosch, Sophie Hospel
Date: 18-03-20
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix

from Objects.big_brain_machine_learning import BigBrainMachineLearning
from calculate_scores import get_score_list


def main(title, train=True, show_confusion=True):
    """Main function that runs the Machine Learning pipeline. Accepts the job title and booleans to indicate wether or
    not to train the data and show the confusion matrix.

    :param title: Title of the job used to pickle the classifier.
    :param train: True if the classifier needs to be trained.
    :param show_confusion: True if the confusion matrix should be shown.
    """
    if train:
        machine_learner = BigBrainMachineLearning("Files/train_set.fasta", title)
        classifier = machine_learner.classifier
    else:
        machine_learner = pickle.load(open(f"Files/{title}.pickle", "rb"))
        classifier = machine_learner.classifier

    test_weighted_sequences = get_score_list(machine_learner.test_sequences)
    accuracy_score = classifier.score(test_weighted_sequences, machine_learner.test_ids)
    print(f"Accuracy Score: {accuracy_score}")

    if show_confusion:
        confusion(classifier, test_weighted_sequences, machine_learner.test_ids)


def confusion(classifier, test_weighted_sequences, test_ids):
    """Shows the confusion matrix as print output and in matplotlib heatmaps.

    :param classifier: The classifier that should be tested.
    :param test_weighted_sequences: The weighted sequences of the test data.
    :param test_ids: Class ids of the test data.
    """
    np.set_printoptions(precision=2)

    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, test_weighted_sequences, test_ids,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


if __name__ == '__main__':
    main("rbf_hydrophobicity", train=True, show_confusion=True)
