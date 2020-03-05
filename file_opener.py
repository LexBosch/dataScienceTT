import plzwerk
from protein_seq_input import ProteinClassification
import pickle


def main():
    data_segments = open_file("train_set.fasta")
    segment_objects = call_new_object(data_segments)
    class_ids = []
    sequences = []

    for single_object in segment_objects:
        class_ids.append(single_object.get_type())
        sequences.append(single_object.get_sequence())

    # new_classifier = BigBrainMachineLearning(class_ids, sequences)

    new_classifier = pickle.load( open( "filename.pickle", "rb" ) )

    benchmark_data_segments = open_file("benchmark_set.fasta")
    benchmark_segment_objects = call_new_object(benchmark_data_segments)
    benchmark_verify_list = []
    benchmark_verify_data = []

    for single_object in benchmark_segment_objects:
        benchmark_verify_list.append(single_object.get_type())
        benchmark_verify_data.append(single_object.get_sequence())

    uglylistplz = plzwerk.get_ugly_list(benchmark_verify_data)
    plx = new_classifier.score(uglylistplz, benchmark_verify_list)
    print(benchmark_verify_list[-100:-1])
    for seq in benchmark_verify_list:
        if seq != "NO_SP":
            print(seq)
    print("asdfasdf")

    return new_classifier, class_ids, uglylistplz, benchmark_verify_list


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