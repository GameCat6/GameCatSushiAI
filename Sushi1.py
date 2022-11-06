import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from lightfm.data import Dataset 
from lightfm.evaluation import auc_score
from scipy import sparse
import csv
import pickle

def feature_to_onehot(max_num,num):
    l2 = list()
    if max_num == 1:
        if num == 1:
            l2.append(1.)
        else:
            l2.append(0.)
        return l2
        exit
    l1 = np.zeros((1,max_num))
    l1[0,num] = 1    
    l1 =  l1[0].tolist()
    return l1

datalistsushi = pd.read_csv(
    "Dataset/sushi3b.5000.10.order",
    sep = ' '
)

datalistsushi=datalistsushi.drop (axis=1,columns=['0','1'])

datauser = pd.read_csv(
    'Dataset/sushi3.udata',
    sep = '\t'
)

data1user = datauser.drop(axis=1,columns = ['ttnf','uid','prefid2',  'regid2',  'e/w2',  'cf'])

i = 1
data2matrixlistuser = list()
for i in range(0,5000):
    data1listuser = list()
    data1listuser = (feature_to_onehot(5000,i) +
                    feature_to_onehot(1,data1user.values[i,0]))
                    # feature_to_onehot(48,data1user.values[i,2]) +
                    # feature_to_onehot(12,data1user.values[i,3]) +
                    # feature_to_onehot(1,data1user.values[i,4]))
    data2matrixlistuser.append(data1listuser)
featurenparr = np.array(data2matrixlistuser)
final_user_features = sparse.csr_matrix(featurenparr)
intermatrix = list()
j = 0
for j in range(0,5000):
    npstrarr1 = np.array(feature_to_onehot(100,datalistsushi.values[j,0]))
    for q in range(1,10):
        npstrarr1 += np.array(feature_to_onehot(100,datalistsushi.values[j,q]))
    intermatrix.append(npstrarr1.tolist())
final_interactions = sparse.coo_matrix(np.array(intermatrix))

interwmatrix = list()
j = 0
for j in range(0,5000):
    npstrarr1 = np.array(feature_to_onehot(100,datalistsushi.values[j,0]))*(q+1)
    for q in range(1,10):
        npstrarr1 += np.array(feature_to_onehot(100,datalistsushi.values[j,q]))*(q+1)
    interwmatrix.append(npstrarr1.tolist())
final_weights = sparse.coo_matrix(np.array(interwmatrix))
file1 = open('C:/Python Programs/SushiRec/SaveAI/my_model/interactions.txt', 'wb')
file2 = open('C:/Python Programs/SushiRec/SaveAI/my_model/features.txt', 'wb')
pickle.dump(final_interactions,file1)
pickle.dump(final_user_features,file2)
file2.close()
file1.close()
model = LightFM(loss='warp')
model.fit(final_interactions,
      sample_weight= final_weights,
      user_features= final_user_features,#user_features,
      epochs=1000,
      verbose = 1)
res = np.argsort(-model.predict(0, np.arange(100)))
# for i in range(0,100):
    # print(i,' : ',res[i])
# train_auc = auc_score(model,
#                        final_interactions,
#                        user_features=final_user_features
#                       ).mean()
file = open('C:/Python Programs/SushiRec/SaveAI/my_model/ModelWarp.txt', 'wb')

pickle.dump(model,file)

file.close()

# print(train_auc)