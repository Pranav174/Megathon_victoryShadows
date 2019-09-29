import re
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
# print(cachedStopWords)

def preprocessing (text):
    text = text.lower()
    text = re.sub(r"(\S)\(", r'\1 (', text)
    regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    text = re.sub(regex, r'\n', text)
    text = text.split('\n')
    output=[]
    for sent in text:
        punct = r"[!”#$%&’,-\./:;<=>^_`{}()\?]"
        sent = re.sub(punct, "   ", sent)
        sent = re.sub(' +', ' ', sent)
        if(sent[-1]==" "):
            sent = sent[0:-1]
        if(sent[0]==" "):
            sent = sent[1:]
        sent = sent.split(" ")
        sent = [word for word in sent if word not in(cachedStopWords)]
        output.append(sent)
    return output

print(preprocessing("HI  is this me wearing a pink shirt? and how do I know when to study 5g "))