def doc_tokenizer(fpath):
    #imports
    import pandas as pd
    from nltk.tokenize import sent_tokenize

    with open(fpath,encoding="utf-8") as f:
        lines = f.readlines()

    lines = ' '.join(lines)
    sents = pd.DataFrame(sent_tokenize(lines))
    sents.rename(columns={0:"text"},inplace=True)

    sents = sents.replace("\n","",regex=True)
    sents.text = sents.text.str.lower()
    sents.text = sents.text.str.replace(r'[^\w\s]+', '',regex=True)

    return sents

def embed(sents):
    import torch
    from torchnlp.word_to_vector import FastText
    from nltk.corpus import stopwords

    stop = stopwords.words('portuguese')
    for idx in range(0,(sents.shape[0])):
        sents.at[idx,"text"] = " ".join([word for word in sents.text.iloc[idx].split() if not word in stop])

    vectors = FastText(language="pt")

    out2 = torch.empty(size=[sents.shape[0],(300+1)])
    for idx in range(0,sents.shape[0]):
        sentence = sents.text.iloc[idx].split(" ")
        temp = vectors[sentence]
        embedding = temp.sum(axis=0)
        out2[idx][1:] = embedding

    return out2

def doc_predict(model,embeddings):
    import numpy as np

    num_sent = embeddings.shape[0]
    new_preds = np.zeros(num_sent)

    for idx in range(0,num_sent):
        new_preds[idx] = model(embeddings[idx,1:]).argmax()

    return new_preds

