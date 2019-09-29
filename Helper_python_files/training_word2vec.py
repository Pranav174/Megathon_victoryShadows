from gensim.models import Word2Vec
import pyemd
import pickle

with open('all_sentences', 'rb') as fp:
    text = pickle.load(fp)


model = Word2Vec(text, size=100, window=5, min_count=5, workers=4) 
'''
set min count to 5 with bigger data
'''

# print(model.wv.vocab)


model.save('megathonWord2Vec.model')


# train_sentence = ["another pair of sentences", "pretty great"]
# for i in range(len(train_sentence)):
#     train_sentence[i] = train_sentence[i].split()
# train_sentence
# len(train_sentence)
# train = True
# if train:
#     model = Word2Vec.load("megathonWord2Vec.model")
#     model.train(train_sentence, total_examples=len(train_sentence), epochs=1)
# model.save('megathonWord2Vec.model')


# print(model.wv.vocab)