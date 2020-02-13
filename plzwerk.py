from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class BigBrainMachineLearning:
    def __init__(self, class_ids: list, sequences: list):
        self.class_ids = class_ids

        self.count_vect = CountVectorizer()

        self.sequences = sequences
        self.classifier = self.train_module()

    def train_module(self):
        # Vult met vectoren met eigenschappen
        x_train_counts = self.count_vect.fit_transform(self.sequences)
        classifier = MultinomialNB().fit(x_train_counts, self.class_ids)

        return classifier

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
