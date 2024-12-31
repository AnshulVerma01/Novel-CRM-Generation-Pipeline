import os
import io
import csv
import sys
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib
import importlib.resources as resources
from functools import reduce
from itertools import combinations
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from lime.lime_tabular import LimeTabularExplainer

from signaturizer import Signaturizer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import pickle
import joblib

import warnings
warnings.filterwarnings('ignore', '.*X does not have valid feature names*', )
warnings.filterwarnings('ignore', '.*DataFrame is highly fragmented*', )
warnings.filterwarnings('ignore', '.*Could not cast to int64*', )
warnings.filterwarnings('ignore', '.*pandas.Int64Index is deprecated*', )
warnings.filterwarnings('ignore', '.*Trying to unpickle estimator*', )
warnings.filterwarnings('ignore', '.*Could not load dynamic library*', )
warnings.filterwarnings('ignore', '.*Ignore above cudart dlerror*', )
warnings.filterwarnings('ignore', '.*tensorflow/stream_executor*', )

FROM_Sign = '/storage/tmpuser/MetGen/Signaturizer/'

#parser = argparse.ArgumentParser()
#parser.add_argument('-i', help='path to smiles.csv, header name: smiles')
#parser.add_argument('-I', help='path to preprocessed_feature_file.pkl')
#parser.add_argument('-O', help='path to output directory')
#args = parser.parse_args()

# # Featurizing and Preprocessing Class

# In[10]:

#print('Reading all classes.')

def Feature_Signaturizer(smiles):
    print('Performing Signaturizer')
    sig_df = pd.DataFrame({'smiles': [smiles]})
    desc=['A','B','C','D','E']
    for dsc in tqdm(desc):
        for i in range(1,6):
#             print('Performing '+dsc+str(i)+' Descriptor Calculation.')
            sign = Signaturizer(dsc+str(i))
            results = sign.predict([smiles])
#             print('Performing '+str(list(pd.DataFrame(results.signature).shape)))
            df=pd.DataFrame(results.signature)
            for clm in list(df.columns):
                df=df.rename(columns={clm:dsc+str(i)+'_'+str(clm)})
            sig_df=pd.concat([sig_df,df],axis = 1)
    return handle_missing_values(sig_df)

def handle_missing_values(Idata):
    print('Processing Missing Values in the generated features.')
    data = Idata.drop(['smiles'],axis=1)
    data = data.replace([np.inf, -np.inf, "", " "], np.nan)
    data = data.replace(["", " "], np.nan)
    for i in data.columns:
        data[i] = data[i].fillna(data[i].mean())
    data['smiles'] = Idata['smiles']
    with open('/storage/tmpuser/MetGen/Signaturizer/'+'preprocessed_feature_file.pkl', 'wb') as f:
        pickle.dump(data, f)    
    return data






# # Getting Predictions Class

# In[12]:


def get_predictions(probs,thresh):
    preds = []
    for prob in probs:
        if prob >= thresh:
            preds.append(1)    
        else:
            preds.append(0)
    return preds


# # Signaturizer Anti Aging Model Class

# In[61]:


class model_sign_aging:
    def __init__(self,test):        
        self.test = test
    def extract_feature(self,model,data):
        F_names = model.feature_names_in_
        return data[F_names]
    def get_labels(self,pred_test): #Getting discrete labels from probability values    
        test_pred = []        
        for i in range(pred_test.shape[0]):
            if(pred_test[i][0]>pred_test[i][1]):
                test_pred.append(0)
            else:
                test_pred.append(1)
        return test_pred        
    def test_model(self):
        test = self.test
        model = joblib.load(FROM_Sign+'AIM_D_Signaturizer_model_svm_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model, test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# # Calculating Prediction Class

# In[99]:


def CanAge_Tox_Predictions(Sig_data):
    
    ################################### Output dataframes ##########################################################################    
    predictions = pd.DataFrame(columns=['smiles','Anti_Aging_Status','Anti_Aging_Prob'])

    predictions['smiles'] = Sig_data['smiles']
    
    ################################## Signaturizer Anti Aging Model #################################################################
    print('Onto Anti-Aging Predictions.')
    m0 = model_sign_aging(Sig_data)
    probs,preds = m0.test_model()
    predictions['Anti_Aging_Prob'] = probs[:,1]
    predictions['Anti_Aging_Status'] = preds    
    
    # Extract only the Anti_Aging_Prob column as a Series
    output_scores = predictions['Anti_Aging_Prob']
    result = output_scores.values[0]
    
    print('Saved CanAge predictions and prediction probabilities')
    #print(result)
    return result

    
# In[43]:
    
             
    
def Predict_Meta(input_data):
    Sig_Input = Feature_Signaturizer(input_data)
    predictions = CanAge_Tox_Predictions(Sig_Input)
    return predictions

# # Testing Data

# In[ ]:


#print('Reading input file -')


# In[ ]:


#with open(args.i) as file:
#   data = file.read().splitlines()


# In[ ]:


#data = pd.DataFrame(data)
#data = data.rename(columns=data.iloc[0]).drop(data.index[0])


#print('Executing the script.')


# In[ ]:

#Predict_Meta("CN(C(=N)N=C(N)N)C")


