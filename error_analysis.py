import nltk
from nltk.tag import tnt
from data_processing import *
# from nltk.corpus import treebank

# initializing training and testing set
#nltk.download('treebank')
#train_data = treebank.tagged_sents()[:10]
#test_data = treebank.tagged_sents()[10:20]
train_data = process_data("train.txt")
dev_data = process_data("dev.txt")
# error analysis
dev_data_ea = process_data_just_tokens("dev.txt")

# test_data = process_data("test.txt")
# print("train data:")
# print(train_data[1:4])
# print("test data:")
# print(test_data[1:4])
# initializing tagger
tnt_tagger = tnt.TnT(N=1000)

# training
tnt_tagger.train(train_data)

# evaluating
predicted = tnt_tagger.tagdata(dev_data_ea)
# print("data")
# print(dev_data_ea)
# print("predicted")
# print(predicted)

wrong_ones_token = []
wrong_ones_sentence = []
for i in range(0, len(predicted)):
    dev = dev_data[i]
    pred = predicted[i]
    flag = 0
    for j in range(0, len(dev)):
        if dev[j] != pred[j]:
            flag = 1
            temp = []
            temp.append(dev[j])
            temp.append(pred[j])
            wrong_ones_token.append(temp)
    temp2 = [dev, pred]
    wrong_ones_sentence.append(temp2)

fp = open('error_token.txt', 'w')
for token_list in wrong_ones_token:
    fp.write(str(token_list))
    # fp.write(", ".join('({}, {})'.format(x[0], x[1]) for x in token_list))
    fp.write("\n")

fp = open('error_sentence.txt', 'w')
for sentence_list in wrong_ones_sentence:
    fp.write(str(sentence_list))
    fp.write("\n\n")