"""
Author: AI experts
Date: 05-03-20
"""

from Objects.Protein_classification import ProteinClassification
from Objects.BBmachinelearning import BigBrainMachineLearning
from calculate_scores import get_score_list
import pickle


def read_and_format_trainset(train):
    """
    This function gets the boolean value of 'train' from the main.
    If the value is False the code will not set the class ids and
    sequences from the train data into an object.
    """

    train_data_segments = open_file("Files/train_set.fasta")
    segment_objects = make_object_from_segments(train_data_segments)
    class_ids, sequences = make_object_from_class_and_sequence(segment_objects)

    if train:
        BigBrainMachineLearning(class_ids, sequences)
    new_classifier = pickle.load(open("Files/filename3.pickle", "rb"))

    benchmark_data_segments = open_file("Files/benchmark_set.fasta")
    benchmark_segment_objects = make_object_from_segments(benchmark_data_segments)
    benchmark_verify_list, benchmark_verify_data = make_object_from_class_and_sequence(benchmark_segment_objects)

    all_scores_list = get_score_list(benchmark_verify_data)
    accuracy_score = new_classifier.score(all_scores_list, benchmark_verify_list)


def open_file(file_name) -> list:
    """
    Gets (path and) filename which to open. Opens the file
    and saves the line into a list. Returns this list.
    :param file_name: string with (path and) filename.
    :return content_file: list with content file.
    """

    content_file = []
    with open(file_name, "r") as file:
        opened_fasta_file = file.read()
        for single_prot_segment in opened_fasta_file.split(">"):
            content_file.append(single_prot_segment)

    return content_file[1:]


def make_object_from_segments(segment_data: list):
    """
    Gets list with data segments. With imported object
    ProteinClassification gets list with objects header,
    sequence and classification from each amino acid in sequence.
    :param segment_data: list with data segments from train or benchmark datafile.
    :return segment_objects: 2D list with objects per segment.
    """

    segment_objects = []
    for single_segment in segment_data:
        head, seq, classifier = single_segment.split("\n")[:-1]
        segment_objects.append(ProteinClassification(head, seq, classifier))

    return segment_objects


def make_object_from_class_and_sequence(segment_objects):
    """
    Gets list with information per segment. Per segment add
    classification to class_ids list and sequence to sequences.

    :param segment_objects: 2D list with objects per segment.
    :return:
    """

    class_ids = []
    sequences = []

    for single_object in segment_objects:
        class_ids.append(single_object.get_type())
        sequences.append(single_object.get_sequence())

    for i in range(len(class_ids)):
        if class_ids[i] != "NO_SP":
            class_ids[i] = "SP"

    for class_id in class_ids: ## deze eigenlijk overbodig want hierboven als geen NO_SP dan SP dus dan heb je 2 values?
        if class_id != "SP" and class_id != "NO_SP":
            print(class_id)

    return class_ids, sequences


if __name__ == '__main__':
    read_and_format_trainset(train=False)
