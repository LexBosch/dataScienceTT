"""Class to store protein data into.

Author: AI experts
Date: 05-03-20
"""


class Protein:
    """This class orders information from header, sequence and classification
    into objects for further use. Splits header for necessarily information.
    """
    def __init__(self, header: str, sequence: str, classification: str):
        """Constructor that stores all the data.

        :param header: Header of a protein sequence.
        :param sequence: Amino acid sequence.
        :param classification: Classification of a protein sequence.
        """
        self.sequence_list = sequence
        self.classification = classification
        header_inf = header.split("|")
        self.uniprot_ac = header_inf[0]
        self.kingdom = header_inf[1]
        self.type = header_inf[2]
        self.partition_no = header_inf[3]
