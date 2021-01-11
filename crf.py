import sklearn_crfsuite
from sklearn_crfsuite import metrics
from data_processing import *

# processing the data
train_data = process_data("train.txt")
dev_data = process_data("dev.txt")

# creating features that will be assigned to every word
def word2features(sent, i):
    word = sent[i][0]
    semtag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'semtag': semtag,
#        'semtag[:2]': semtag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        semtag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:semtag': semtag1,
#            '-1:semtag[:2]': semtag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        semtag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:semtag': semtag1,
#            '+1:semtag[:2]': semtag1[:2],
        })
    else:
        features['EOS'] = True

    return features

# turning the sentences into feature sequences
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2semtags(sent):
    return [semtag for token, semtag in sent]

def sent2tokens(sent):
    return [token for token, semtag in sent]

print(sent2features(dev_data[1])[1])

# training data for tokens and tags
X_train = [sent2features(s) for s in train_data]
Y_train = [sent2semtags(s) for s in train_data]

#dev data for tokens and tags
X_dev = [sent2features(s) for s in dev_data]
Y_dev = [sent2semtags(s) for s in dev_data]

# training the algorithm
crf = sklearn_crfsuite.CRF()
crf.fit(X_train, Y_train)

Y_pred = crf.predict(X_train)
print('Accuracy on the train set = {}\n'.format(metrics.flat_accuracy_score(Y_train, Y_pred)))
Y_pred_dev = crf.predict(X_dev)
print('Accuracy on the dev set = {}\n'.format(metrics.flat_accuracy_score(Y_dev, Y_pred_dev)))
print('Classification report of the dev set = {}\n'.format(metrics.flat_classification_report(Y_dev, Y_pred_dev)))
# Playing around with the L1 and L2 regularization parameters might help give us a better performance on the test set and
# prevent overfitting. (https://towardsdatascience.com/pos-tagging-using-crfs-ea430c5fb78b)