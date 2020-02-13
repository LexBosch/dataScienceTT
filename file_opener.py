from protein_seq_input import ProteinClassification
from plzwerk import BigBrainMachineLearning
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    data_segments = open_file("train_set.fasta")
    segment_objects = call_new_object(data_segments)
    print("ja")

    varify_list = []
    varify_data = []
    for single_object in segment_objects:
        varify_list.append(single_object.get_type())
        varify_data.append(single_object.get_sequence())

    new_classifier = BigBrainMachineLearning(varify_list, varify_data)

    benchmark_data_segments = open_file("benchmark_set.fasta")
    benchmark_segment_objects = call_new_object(benchmark_data_segments)
    benchmark_varify_list = []
    benchmark_varify_data = []
    for single_object in benchmark_segment_objects:
        benchmark_varify_list.append(single_object.get_type())
        benchmark_varify_data.append(single_object.get_sequence())



    X_new_counts = new_classifier.get_count_vect().transform(benchmark_varify_data)

    plx = new_classifier.get_classifier().score(X_new_counts, benchmark_varify_list)


    print("asdf")





def open_file(file_name) -> list:
    dataset_notanymore_empty_list = []
    with open(file_name, "r") as file:
        opened_fasta_file = file.read()
        for single_prot_segment in opened_fasta_file.split(">"):
            dataset_notanymore_empty_list.append(single_prot_segment)
    return dataset_notanymore_empty_list[1:]


def call_new_object(segment_data: list):
    segment_objects = []
    for single_segment in segment_data:
        head, seq, classif = single_segment.split("\n")[:-1]
        segment_objects.append(ProteinClassification(head, seq, classif))
    return segment_objects


if __name__ == '__main__':
    main()