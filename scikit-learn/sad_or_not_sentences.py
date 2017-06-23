#!/bin/python

print('Loading modules and data')
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

with open('sentences/sad.txt') as fin:
    sad = fin.readlines()
with open('sentences/not_sad.txt') as fin:
    not_sad = fin.readlines()

print('Pre-processing data')
for x in range(0, len(sad)):
    sad[x] = sad[x].lower()
for y in range(0, len(not_sad)):
    not_sad[y] = not_sad[y].lower()

X_train = np.array(sad + not_sad)
y_train_text = [["sad"]]*len(sad) + [["not sad"]]*len(not_sad)

X_test = ['You would really hate being me',
         'I am really lucky to be born in this era',
         'I gave up everything and jumped',
         'I feel I am finally done with everything',
         'It was awful, the way he let go of it',
         'I am glad I could be of help',
         'I love candies',
         'Mate, you outperformed everbody',
         'Everything was going great until she disappeared',
         'What did I do that I deserved this?',
         'My past was really hard',
         'I hated my childhood',
         'I like that tv series very much',
         'My brother is an idiot, he ate my share of candies',
         'Am I really that terrible',
         'He failed all his subjects',
         'Microsoft is making another toolset']

for z in range(0, len(X_test)):
    X_test[z] = X_test[z].lower()

X_test = np.array(X_test)

print('Training')
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)
print('Predicting\n')
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print('{0} => {1}'.format(item, ', '.join(labels)))
