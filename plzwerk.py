from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from time import time
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from math import fsum
import pickle


def get_ugly_list(sequences):
    arbitrair = {"R": -4.5,
                      "K": -3.9,
                      "N": -3.5,
                      "D": -3.5,
                      "Q": -3.5,
                      "E": -3.5,
                      "H": -3.2,
                      "P": -1.6,
                      "Y": -1.3,
                      "W": -0.9,
                      "S": -0.8,
                      "T": -0.7,
                      "G": -0.4,
                      "A": 1.8,
                      "M": 1.9,
                      "C": 2.5,
                      "F": 2.8,
                      "L": 3.8,
                      "V": 4.2,
                      "I": 4.5}
    ja = []
    groote_van_segment = 10
    for single_sequence in sequences:
        ja1 = []
        single_sequence = single_sequence[0:40]
        for i in range(0, (len(single_sequence) - groote_van_segment)):
            sub_seq = single_sequence[i:i + groote_van_segment]
            total_score = round(sum(arbitrair[amino] for amino in sub_seq), 2)
            ja1.append(total_score)
        ja.append(ja1)
    return ja

class BigBrainMachineLearning:
    def __init__(self, class_ids: list, sequences: list):
        self.class_ids = class_ids

        self.count_vect = CountVectorizer()

        self.sequences = sequences

        self.classifier = self.train_module()





    def train_module(self):
        # Vult met vectoren met eigenschappen
        x_train_counts = self.count_vect.fit_transform(self.sequences)

        ja = get_ugly_list(self.sequences)

        time()
        plzwerk = SVC(kernel="linear")
        plzwerk.fit(ja, self.class_ids)
        time()
        with open('filename3.pickle', 'wb') as handle:
            pickle.dump(plzwerk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_classifier(self):
        return self.classifier

    def get_count_vect(self):
        return self.count_vect


#
#
#
#
# train = ["aaaaaaaaaaa",
#                   "aaaaaaaaaa",
#                   "bbbbbbbbbb",
#                   "bbbbbbbbbbbbbbbbbbbbbbbbbb",
#                   "bbbbbbbbb"]
#
# class_train = ["a", "a", "b", "b", "b"]
#
# test = ["aaaaaaaaaaa",
#                   "aaaaaaaaaa",
#                   "bbbbbbbbbb",
#                   "bbbbbbbbbb",
#                   "bbbbbbbbb"]
#
# class_test = ["nosp", "nosp", "sp", "sp", "sp"]
#
# ####################################
#
#
# # maakt dictionary van eigenschappen van de mogelijke data

#
# kleine_test_data = ["aaaaaaaaaa", "bbbbbbbbbbb", "aaaaaaaaaaaaaaaaaaaaaaaaaa"]
#
# X_new_counts = count_vect.transform(kleine_test_data)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#
# predicted = clf.predict(X_new_tfidf)
#
# print("reeee")
#
#
#
#
# svc = svm.SVC(kernel='rbf')
#
# #
# # count_vect = CountVectorizer()
# x_train = count_vect.fit_transform(train)
#
# svc.fit(train, class_train)
#
# #
# # tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# # X_train_tf = tf_transformer.transform(X_train_counts)
#
#
# predicted = svc.predict(test)
# score = svc.score(test, class_test)
#
# print('============================================')
# print('\nScore ', score)
# print('\nResult Overview\n', metrics.classification_report(class_test, predicted))
# print('\nConfusion matrix:\n', metrics.confusion_matrix(class_test, predicted))
#
# ##########################################
# cmap, cmapMax = plt.cm.RdYlBu, ListedColormap(['#FF0000', '#0000FF'])
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# h = 0.3
# x_min, x_max = train[:, 0].min() - .3, train[:, 0].max() + .3
# y_min, y_max = train[:, 1].min() - .3, train[:, 1].max() + .3
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# if hasattr(svc, "decision_function"):
#     Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
# else:
#     Z = svc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
# Z = Z.reshape(xx.shape)
# ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.7)
#
# # Plot also the training points
# ax.scatter(train[:, 0], train[:, 1], c=class_train, cmap=cmapMax)
# # and testing points
# ax.scatter(test[:, 0], test[:, 1], c=class_test, cmap=cmapMax, alpha=0.5)
#
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xticks(())
# ax.set_yticks(())
# plt.title(str(score))
#
# plt.show()
#
