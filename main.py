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
tnt_tagger = tnt.TnT(N=1500)

# training
tnt_tagger.train(train_data)

# evaluating
a = tnt_tagger.evaluate(dev_data)

print("Accuracy of TnT Tagging : ", a)
