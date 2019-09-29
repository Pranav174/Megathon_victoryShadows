import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import networkx as nx
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


model = Word2Vec.load('megathonWord2Vec.model')
model.init_sims(replace=True)
def preprocessing (text):
    text = text.lower()
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
                sent = sent.split(" ")
                sent = [word for word in sent if word not in(cachedStopWords)]
                output.append(sent)
    return output

def sen_preprocessing(sentence):
    text=sentence
    text = text.lower()
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
                sent = sent.split(" ")
                sent = [word for word in sent if word not in(cachedStopWords)]
                output.extend(sent)
    return output

topn=10
dimention=100
vocab = model.wv.vocab
def rank_sentences(essay):
    x=len(essay)
    sentence_embedding = np.zeros((len(essay),dimention))
    for i,sent in enumerate(essay):
        found = 0
        for token in sent:
            if token in vocab:
                found+=1
                sentence_embedding[i] = np.add(sentence_embedding[i],model.wv[token])
        if found:
            sentence_embedding[i]/found
    sim_mat = np.zeros((x,x))
    sim_mat = cosine_similarity(sentence_embedding, sentence_embedding)
    for i in range(x):
        sim_mat[i][i]=0
    final = []
    # nx_graph = nx.from_numpy_array(sim_mat)
    # scores = nx.pagerank(nx_graph, max_iter=10)
    for i in range(min(len(essay),6)):
        final.extend(essay[i])
    return final

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

fulltext = pd.read_csv("fulltext.csv")
abstracts = pd.read_csv("summaries.csv")
abstracts = abstracts[:10]
fulltext = fulltext[:100]
m = len(fulltext["paper_text"])
n = len(abstracts["abstract"])
similarity_matrix = np.zeros((n,m))

def avg_emd(text):
    ans = np.zeros(dimention)
    found = 0
    for token in text:
        if token in vocab:
            found+=1
            ans = np.add(ans,model.wv[token])
    if found:
        ans/found
    return ans

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def another(x):
    return x / (1+abs(x))

tqdm.pandas()
print("Pre-Processing Text")
fulltext["processed"] = fulltext.progress_apply(lambda x: preprocessing(x['paper_text']),axis=1)
print("Finding patterns in Text")
fulltext["top_sentences"] = fulltext.progress_apply(lambda x: rank_sentences(x['processed']),axis=1)
print("Pre-processing Abstracts")
abstracts["processed"] = abstracts.progress_apply(lambda x: sen_preprocessing(x['abstract']),axis=1)

print('Generating similarity matrix')
for i in tqdm(range(n)):
    for j in tqdm(range(m)):
        # similarity_matrix[i][j]=model.n_similarity(abstracts.loc[i, "processed"], fulltext.loc[j,"top_sentences"])
        # similarity_matrix[i][j]=cosine_similarity(avg_emd(abstracts.loc[i, "processed"]).reshape(1,100), avg_emd(fulltext.loc[j,"top_sentences"]).reshape(1,100))[0][0]
        similarity_matrix[i][j]=model.wmdistance(abstracts.loc[i, "processed"], fulltext.loc[j,"top_sentences"])


for i in range(n):
    minn = min(similarity_matrix[i])
    maxx = max(similarity_matrix[i])
    differnce = maxx-minn
    for j in range(m):
        # similarity_matrix[i][j] = (differnce-(similarity_matrix[i][j]-minn))/differnce
        # similarity_matrix[i][j] = (similarity_matrix[i][j]-minn)/differnce
        # similarity_matrix[i][j] = (((1-sigmoid(similarity_matrix[i][j]))-0.25)*2)+0.5
        similarity_matrix[i][j] = (similarity_matrix[i][j]-minn)/differnce
        similarity_matrix[i][j] = 1-(2*another(similarity_matrix[i][j]))



pd.DataFrame(similarity_matrix).to_csv("similarity_matrix.csv", header=False,index=False)