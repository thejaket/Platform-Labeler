import copy

import torch
from torch import nn
from torchnlp.word_to_vector import *
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torchmetrics import PrecisionRecallCurve
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
from datetime import datetime
from skorch import NeuralNetClassifier

from network_trainer import *
from tokenizer import *
import os

class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.labels = labels
        self.embeddings = embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        embeddings = self.embeddings[idx]
        return label, embeddings

def main():
    filename = "complete_labels.csv"
    #filename = "platform_8_21.csv"

    of_interest = ["economy","education","fluff","healthcare","justice",
                   "service_other","infrastructure","management","security"]
    #of_interest=["security"]

    #Load, clean sentence labels, keeping only those that have been read by multiple
    #readers (tagged as 1 on "complete")
    plat = pd.read_csv(filename, encoding="UTF-8")
    plat = plat.iloc[0:(plat.shape[0] - 1)]

    plat = plat.replace("\n", "", regex=True)
    plat.text = plat.text.str.lower()
    plat.text = plat.text.str.replace(r'[^\w\s]+', '', regex=True)

    mask = plat.complete == 1

    plat = plat.loc[mask]

    #PREPROCESSING
    #import list of portuguese stopwords, remove from sentences
    stop = stopwords.words('portuguese')

    for idx in range(0, (plat.shape[0])):
        #plat.at[idx, "text"] = " ".join([word for word in plat.text.iloc[idx].split() if not word in stop])
        plat.text.iloc[idx] = " ".join([word for word in plat.text.iloc[idx].split() if not word in stop])

    for topic in of_interest:
        plat[topic] = plat[topic].astype(int)
        plat[topic] = plat[topic].astype(int)

    #Select pre-trained word embedding model (value set by user)
    languageModel = 'fasttext'

    if languageModel=='glove':
        vectors = GloVe()
    elif languageModel=='fasttext':
        vectors = FastText(language="pt")
    elif languageModel=='bmemp':
        vectors = BPEmb(language="pt")

    #Find word embeddings, combine to create sentence embedding (vector addition)
    outs = torch.empty(size=[plat.shape[0],(300+len(of_interest))])
    for idx in range(0,plat.shape[0]):
        for topic in range(0,len(of_interest)):
            outs[idx][topic] = plat[of_interest[topic]].iloc[idx]

        sentence = plat.text.iloc[idx].split(" ")
        temp = vectors[sentence]
        embedding = temp.sum(axis=0)
        outs[idx][len(of_interest):] = embedding

    #Split data into train, test sets
    train_set, test_set = train_test_split(outs,test_size=0.2)

    #Create container for the models, select batch size and number of Epochs
    #Right now arbitrary values, final version will inlcude code for tuning
    #these hyperparameters
    models = []
    batch_size = 64
    epochs = 50

    #Run through each label, train a binary classifier for each one
    #Future version will also train single multi-label classifier and compare performance
    for i in range(0,len(of_interest)):
        pos_weight = 1-sum(train_set[:,0]==1)/(sum(train_set[:,0]==1)+sum(train_set[:,0]==0))
        neg_weight = 1-sum(train_set[:, 0] == 0) / (sum(train_set[:, 0] == 1) + sum(train_set[:, 0] == 0))

        weight_vec = copy.deepcopy(train_set[:,0])
        weight_vec[weight_vec==1] = pos_weight
        weight_vec[weight_vec == 0] = neg_weight

        #Pull weighted random sample to oversample positive cases
        #All labels are minority classes, especially security which has around 2% positive cases
        sampler = WeightedRandomSampler(weight_vec,train_set.shape[0])
        train_dataset = CustomDataset(train_set[:, len(of_interest):], train_set[:, i:(i+1)])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,sampler=sampler)

        test_dataset = TensorDataset(test_set[:, len(of_interest):], test_set[:, i:(i+1)])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        #Train, validate neural network (see other code for function definitions)
        model = network(train_dataloader, test_dataloader,epochs)
        models.append(model)
        validate(model, test_dataset, test_dataloader)

    #CHANGE DIRECTORY
    files = os.listdir("txt")

    #Read in txt files
    files = [name for name in files if ".txt" in name]

    testnumber = len(files)
    #testnumber = 10

    #Create empty prediction dataframe
    bigpred = pd.DataFrame(columns=of_interest)
    bigpred = bigpred.assign(name=pd.Series(),sentence=pd.Series())
    for idx in range(0,testnumber):
        filepath = f"txt/{files[idx]}"

        #Tokenize new platform, generate sentence embeddings
        sents = doc_tokenizer(filepath)
        embeddings = embed(sents)

        temp = pd.DataFrame(columns=of_interest)
        temp = temp.assign(sentence= sents.text)
        temp = temp.assign(name= files[idx])

        #Predict labels for new documents
        for i in range(0,len(models)):
            preds = doc_predict(models[i],embeddings)
            temp[of_interest[i]] = preds

        bigpred = pd.concat([bigpred,temp])

        os.system('cls')
        print(f"{(idx+1)/testnumber*100}%")

        outfile = f"{datetime.today().strftime('%Y-%m-%d')}_{languageModel}.csv"
        if idx==0:
            bigpred.to_csv(outfile, encoding='utf-8',index=False)
            bigpred = pd.DataFrame(columns=bigpred.columns)

        elif idx%100==0:
            bigpred.to_csv(outfile, encoding='utf-8', mode='a',index=False,header=False)
            bigpred = pd.DataFrame(columns=bigpred.columns)

        elif idx==(testnumber-1):
            bigpred.to_csv(outfile, encoding='utf-8', mode='a', index=False, header=False)

main()
