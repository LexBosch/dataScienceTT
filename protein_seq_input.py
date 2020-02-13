class ProteinClassification:
    def __init__(self, header: str, sequence: str, classification: str):
        self.sequence_list = sequence
        self.classification = classification
        header_inf = header.split("|")
        self.uniprot_ac = header_inf[0]
        self.kingdom = header_inf[1]
        self.type = header_inf[2]
        self.partition_no = header_inf[3]

    def get_type(self):
        return self.type

    def get_sequence(self):
        return self.sequence_list
