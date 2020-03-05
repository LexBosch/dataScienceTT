"""
Author: AI experts
Date: 05-03-20
"""

import plzwerk
from Objects.protein_seq_input import ProteinClassification
from Objects.BBmachinelearning import BigBrainMachineLearning
import pickle


def read_and_format_trainset(train):
    """

    :return:
    """

    data_segments = open_file("Files/train_set.fasta")
    segment_objects = make_object_from_segments(data_segments)
    class_ids, sequences = make_object_from_class_and_sequence(segment_objects)

    if train:
        BigBrainMachineLearning(class_ids, sequences)
    new_classifier = pickle.load(open("Files/filename3.pickle", "rb"))

    benchmark_data_segments = open_file("Files/benchmark_set.fasta")
    benchmark_segment_objects = make_object_from_segments(benchmark_data_segments)

    benchmark_verify_list = []
    benchmark_verify_data = []

    for single_object in benchmark_segment_objects:
        benchmark_verify_list.append(single_object.get_type())
        benchmark_verify_data.append(single_object.get_sequence())

    for i in range(len(benchmark_verify_list)):
        if benchmark_verify_list[i] != "NO_SP":
            benchmark_verify_list[i] = "SP"

    for class_id in benchmark_verify_list:
        if class_id != "SP" and class_id != "NO_SP":
            print(class_id)

    uglylistplz = plzwerk.get_ugly_list(benchmark_verify_data)
    plx = new_classifier.score(uglylistplz, benchmark_verify_list)
    print("asdfasdf")

    return new_classifier, class_ids, uglylistplz, benchmark_verify_list


def open_file(file_name) -> list:
    dataset_notanymore_empty_list = []
    with open(file_name, "r") as file:
        opened_fasta_file = file.read()
        for single_prot_segment in opened_fasta_file.split(">"):
            dataset_notanymore_empty_list.append(single_prot_segment)
    return dataset_notanymore_empty_list[1:]


def make_object_from_segments(segment_data: list):
    segment_objects = []
    for single_segment in segment_data:
        head, seq, classif = single_segment.split("\n")[:-1]
        segment_objects.append(ProteinClassification(head, seq, classif))

    return segment_objects


def make_object_from_class_and_sequence(segment_objects):
    class_ids = []
    sequences = []

    for single_object in segment_objects:
        class_ids.append(single_object.get_type())
        sequences.append(single_object.get_sequence())

    for i in range(len(class_ids)):
        if class_ids[i] != "NO_SP":
            class_ids[i] = "SP"

    for class_id in class_ids:
        if class_id != "SP" and class_id != "NO_SP":
            print(class_id)

    return class_ids, sequences


if __name__ == '__main__':
    read_and_format_trainset(train=False)
