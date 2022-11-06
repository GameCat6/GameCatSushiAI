import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from lightfm.data import Dataset 
from lightfm.evaluation import reciprocal_rank, auc_score,precision_at_k
from scipy import sparse
import csv
import pickle

file = open('C:/Python Programs/SushiRec/SaveAI/my_model/ModelWarp.txt', 'rb')
file1 = open('C:/Python Programs/SushiRec/SaveAI/my_model/interactions.txt', 'rb')
file2 = open('C:/Python Programs/SushiRec/SaveAI/my_model/features.txt', 'rb')
model = pickle.load(file)
interactions = pickle.load(file1)
user_features = pickle.load(file2)
train_auc = auc_score(model,
                      interactions,
                      user_features=user_features
                     ).mean()
print('auc_score :',train_auc)
train_auc = reciprocal_rank(model,
                      interactions,
                      user_features=user_features
                     ).mean()
print('reciprocal_rank :', train_auc)
train_auc = precision_at_k(model,
                      interactions,
                      user_features=user_features,
                      k= 100
                     ).mean()
print('precision_at_k :',train_auc)
print(np.argsort(-model.predict(0, np.arange(100)))[:3])
print(interactions.todense()[0])