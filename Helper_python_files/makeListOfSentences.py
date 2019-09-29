import spacy 
import re
import pickle
from tqdm import tqdm
import pandas as pd

def preprocessing(text):
    text = re.sub(r"(\S)\(", r'\1 (', text)
    regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    text = re.sub(regex, r'\n', text)
    text = text.split('\n')
    output=[]
    for sent in text:
        if len(sent):
            punct = r"[!”#$%&’,-\./:;<=>^_`{}()\?]"
            sent = re.sub(punct, "   ", sent)
            sent = re.sub(' +', ' ', sent)
            if(sent[-1]==" "):
                sent = sent[0:-1]
            if len(sent):
                if(sent[0]==" "):
                    sent = sent[1:]
                output.append(sent.split(" "))
    return output

allSentences = []

fulltext = pd.read_csv("test_fulltext.csv")
abstracts = pd.read_csv("test_summaries.csv")
print(len(allSentences))

for sentence in tqdm(fulltext["paper_text"]):
    allSentences.extend(preprocessing(sentence))
print(len(allSentences))

for sentence in tqdm(abstracts["abstract"]):
    allSentences.extend(preprocessing(sentence))

print(len(allSentences))

with open('all_sentences', 'wb') as fp:
    pickle.dump(allSentences, fp)