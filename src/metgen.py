import os
import sys
import random
import csv
import warnings
import io
import argparse
import importlib
import importlib.resources as resources
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from time import time

from utils import Molecule_Dataset
from tdc.oracles import Oracle
from tdc import Evaluator

from functools import reduce

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from lime.lime_tabular import LimeTabularExplainer

from signaturizer import Signaturizer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, Draw
import joblib
from rdkit.Chem.Draw import IPythonConsole
from PIL import Image

from Sig import Predict_Meta

from random import shuffle
torch.manual_seed(4)
np.random.seed(2)
random.seed(1)
torch.manual_seed(1)
np.random.seed(2)

warnings.filterwarnings('ignore', '.*X does not have valid feature names*', )
warnings.filterwarnings('ignore', '.*DataFrame is highly fragmented*', )
warnings.filterwarnings('ignore', '.*Could not cast to int64*', )
warnings.filterwarnings('ignore', '.*pandas.Int64Index is deprecated*', )
warnings.filterwarnings('ignore', '.*Trying to unpickle estimator*', )
warnings.filterwarnings('ignore', '.*Could not load dynamic library*', )
warnings.filterwarnings('ignore', '.*Ignore above cudart dlerror*', )
warnings.filterwarnings('ignore', '.*tensorflow/stream_executor*', )



#INPUT_FILE = input("Please enter the SMILES input filename : ")
##the file name is "filtered_smiles.txt"



def smiles2mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol 

## input: smiles, output: word lst;  
def smiles2word(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None 
    word_lst = []


    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques_smiles = []
    for clique in cliques:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=False) ### Changed to False
        cliques_smiles.append(clique_smiles)
    atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    return cliques_smiles + atom_not_in_rings_list 


## Output: Ring structures and atom
all_vocabulary_file = "/storage/tmpuser/MetGen/data/substructure.txt"
#INPUT_FILE = input("Please enter the SMILES input filename : ")
#rawdata_file = "/storage/tmpuser/MetGen/data/" + INPUT_FILE
rawdata_file = "/storage/tmpuser/MetGen/data/filtered_smiles.txt"
select_vocabulary_file = "/storage/tmpuser/MetGen/data/vocabulary.txt"


if not os.path.exists(all_vocabulary_file):
	with open(rawdata_file) as fin:
		lines = fin.readlines()[0:]
		smiles_lst = [line.strip().strip('"') for line in lines]
	word2cnt = defaultdict(int)
	for smiles in tqdm(smiles_lst):
		word_lst = smiles2word(smiles)
		for word in word_lst:
			word2cnt[word] += 1
	word_cnt_lst = [(word,cnt) for word,cnt in word2cnt.items()]
	word_cnt_lst = sorted(word_cnt_lst, key=lambda x:x[1], reverse = True)

	with open(all_vocabulary_file, 'w') as fout:
		for word, cnt in word_cnt_lst:
			fout.write(word + '\t' + str(cnt) + '\n')
else:
	with open(all_vocabulary_file, 'r') as fin:
		lines = fin.readlines()
		word_cnt_lst = [(line.split('\t')[0], int(line.split('\t')[1])) for line in lines]


word_cnt_lst = list(filter(lambda x:x[1]>10, word_cnt_lst))
print(len(word_cnt_lst))

with open(select_vocabulary_file, 'w') as fout:
	for word, cnt in word_cnt_lst:
		fout.write(word + '\t' + str(cnt) + '\n')


### Filtering the smiles based on validity
from chemutils import is_valid, logp_modifier, smiles2graph, vocabulary, smiles2feature
from module import GCN
from chemutils import * 
from inference_utils import *


clean_smiles_database = "/storage/tmpuser/MetGen/data/vocab_clean.txt"

with open(rawdata_file, 'r') as fin:
	lines = fin.readlines()[0:]
smiles_lst = [i.strip().strip('"') for i in lines]

clean_smiles_lst = []
for smiles in tqdm(smiles_lst):
	if is_valid(smiles):
		clean_smiles_lst.append(smiles)
clean_smiles_set = set(clean_smiles_lst)
with open(clean_smiles_database, 'w') as fout:
	for smiles in clean_smiles_set:
		fout.write(smiles + '\n')
  


#from random import shuffle 
#torch.manual_seed(4) 
#np.random.seed(2) 


device = 'cpu'
#clean_smiles_database = "/storage/tmpuser/MetGen/data/vocab_clean.txt"
with open(clean_smiles_database, 'r') as fin:
	lines = fin.readlines()

shuffle(lines)
lines = [line.strip() for line in lines]
N = int(len(lines) * 0.9)   ###starting 90% smiles used for training and rest for validation
train_data = lines[:N]
valid_data = lines[N:]



training_set = Molecule_Dataset(train_data)
valid_set = Molecule_Dataset(valid_data)
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}
# exit() 


def collate_fn(batch_lst):
	return batch_lst

train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)

gnn = GCN(nfeat = 50, nhid = 100, num_layer = 3).to(device)
print('GNN is built!')
# exit() 

cost_lst = []
valid_loss_lst = []
epoch = 5 
every_k_iters = 5000
save_folder = "/storage/tmpuser/MetGen/save_model/GNN_epoch_" 
err_smi_t=[]
err_smi_v=[]
for ep in tqdm(range(epoch)):
	for i, smiles in tqdm(enumerate(train_generator)):
		### 1. training
		smiles = smiles[0]
		try:
			node_mat, adjacency_matrix, idx, label = smiles2feature(smiles) ### smiles2feature: only mask leaf node
		except:
			err_smi_t.append(smiles)
			continue
		# idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
		node_mat = torch.FloatTensor(node_mat).to(device)
		adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
		label = torch.LongTensor([label]).view(-1).to(device)
		# print('label', label)
		cost = gnn.learn(node_mat, adjacency_matrix, idx, label)
		cost_lst.append(cost)

		#### 2. validation 
		if i % every_k_iters == 0:
			gnn.eval()
			valid_loss, valid_num = 0,0 
			for smiles in valid_generator:
				smiles = smiles[0]
				try:
					node_mat, adjacency_matrix, idx, label = smiles2feature(smiles)
				except:
					err_smi_v.append(smiles)
					continue
				node_mat = torch.FloatTensor(node_mat).to(device)
				adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
				label = torch.LongTensor([label]).view(-1).to(device)
				cost, _ = gnn.infer(node_mat, adjacency_matrix, idx, label)
				valid_loss += cost
				valid_num += 1
			valid_loss = valid_loss / valid_num
			valid_loss_lst.append(valid_loss)
			file_name = save_folder + str(ep) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
			torch.save(gnn, file_name)
			gnn.train()

# Open file for writing error SMILES in training data
with open('/storage/tmpuser/MetGen/data/error_smiles_training.txt', 'w') as f:
    for item in err_smi_t:
        f.write("%s\n" % item)

# Open file for writing error SMILES in validation data
with open('/storage/tmpuser/MetGen/data/error_smiles_validation.txt', 'w') as f:
    for item in err_smi_v:
        f.write("%s\n" % item)


##generation of new SMILES with the average, novelty and Diversity scores of the whole set of SMILES

def optimization(start_smiles_lst, gnn, oracle, oracle_num, oracle_name, generations, population_size, lamb, topk, epsilon, result_pkl):
	smiles2score = dict() ### oracle_num
	def oracle_new(smiles):
		if smiles not in smiles2score:
			value = oracle(smiles) 
			smiles2score[smiles] = value 
		return smiles2score[smiles] 
	trace_dict = dict() 
	existing_set = set(start_smiles_lst)  
	current_set = set(start_smiles_lst)
	average_f = np.mean([oracle_new(smiles) for smiles in current_set])
	f_lst = [(average_f, 0.0)]
	idx_2_smiles2f = {}
	smiles2f_new = {smiles:oracle_new(smiles) for smiles in start_smiles_lst} 
	idx_2_smiles2f[-1] = smiles2f_new, current_set 
	for i_gen in tqdm(range(generations)):
		next_set = set()
		for smiles in current_set:
			smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)

			for smi in smiles_set:
				if smi not in trace_dict:
					trace_dict[smi] = smiles ### ancestor -> offspring 
			next_set = next_set.union(smiles_set)
		# next_set = next_set.difference(existing_set)   ### if allow repeat molecule  
		smiles_score_lst = oracle_screening(next_set, oracle_new)  ###  sorted smiles_score_lst 
		#print(smiles_score_lst[:5], "Oracle num", len(smiles2score))

		# current_set = [i[0] for i in smiles_score_lst[:population_size]]  # Option I: top-k 
		current_set,_,_ = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb) 	# Option II: DPP
		existing_set = existing_set.union(next_set)

		# save 
		smiles2f_new = {smiles:score for smiles,score in smiles_score_lst} 
		idx_2_smiles2f[i_gen] = smiles2f_new, current_set 
		pickle.dump((idx_2_smiles2f, trace_dict), open(result_pkl, 'wb'))

		#### compute f-score
		score_lst = [smiles2f_new[smiles] for smiles in current_set] 
		average_f = np.mean(score_lst)
		std_f = np.std(score_lst)
		f_lst.append((average_f, std_f))
		str_f_lst = [str(i[0])[:5]+'\t'+str(i[1])[:5] for i in f_lst]
		with open("/storage/tmpuser/MetGen/result/" + oracle_name + "_f_t.txt", 'w') as fout:
			fout.write('\n'.join(str_f_lst))
		if len(smiles2score) > oracle_num: 
			break 

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--oracle_num', type=int, default=1500)
	parser.add_argument('--oracle_name', type=str, default="canage", choices=['jnkgsk', 'qedsajnkgsk', 'qed', 'jnk', 'gsk', 'canage'])	
	parser.add_argument('--generations', type=int, default=50)	
	parser.add_argument('--population_size', type=int, default=20)	
	args, unknown = parser.parse_known_args()  #args = parser.parse_args()  (changed)
  

	oracle_num = args.oracle_num 
	oracle_name = args.oracle_name 
	generations = args.generations 
	population_size = args.population_size

	start_smiles_lst = ['CN(C(=N)N=C(N)N)C']  ## Metformin SMILES
	qed = Oracle('qed')
	sa = Oracle('sa')
	jnk = Oracle('JNK3')
	gsk = Oracle('GSK3B')
	logp = Oracle('logp')
	#canage = Predict_Meta('canage')
	mu = 2.230044
	sigma = 0.6526308
	def normalize_sa(smiles):
		sa_score = sa(smiles)
		mod_score = np.maximum(sa_score, mu)
		return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.)) 


	if oracle_name == 'jnkgsk':
		def oracle(smiles):
			return np.mean((jnk(smiles), gsk(smiles)))
	elif oracle_name == 'qedsajnkgsk':
		def oracle(smiles):
			return np.mean((qed(smiles), normalize_sa(smiles), jnk(smiles), gsk(smiles))) 
	elif oracle_name == 'qed':
		def oracle(smiles):
			return qed(smiles) 
	elif oracle_name == 'jnk':
		def oracle(smiles):
			return jnk(smiles)
	elif oracle_name == 'gsk':
		def oracle(smiles):
			return gsk(smiles) 
	elif oracle_name == 'logp':
		def oracle(smiles):
			return logp(smiles)
	elif oracle_name == 'canage':
		def oracle(smiles):
			return Predict_Meta(smiles)

	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = 'cpu' ## cpu is better 
	
	# Define the folder where the files are located
	folder_path = "/storage/tmpuser/MetGen/save_model/"

	# Initialize variables to hold the highest episode number and lowest validation loss
	max_ep = 0
	min_loss = float('inf')
	best_file = ""

	# Loop through all files in the folder
	for filename in os.listdir(folder_path):
	    # Check if the filename matches the pattern
	    if "GNN_epoch_" in filename and "_validloss_" in filename and filename.endswith(".ckpt"):
	        # Extract the episode number and validation loss from the filename
	        parts = filename.split("_")
	        ep = int(parts[2])
	        loss = float(parts[4][:7])
	        # Check if this file has a higher episode number or a lower validation loss than the current best file
	        if ep > max_ep or (ep == max_ep and loss < min_loss):
	            max_ep = ep
	            min_loss = loss
	            best_file = filename

	# Print the name of the best file
	print(best_file)
	
	#model_ckpt = "/storage/tmpuser/MetGen/save_model/GNN_epoch_4_validloss_1.80103.ckpt"
	model_ckpt = "/storage/tmpuser/MetGen/save_model/" + best_file
	gnn = torch.load(model_ckpt)
	gnn.switch_device(device)

	result_pkl = "/storage/tmpuser/MetGen/result/" + oracle_name + ".pkl"
	optimization(start_smiles_lst, gnn, oracle, oracle_num, oracle_name,
						generations = generations, 
						population_size = population_size, 
						lamb=2, 
						topk = 5, 
						epsilon = 0.7, 
						result_pkl = result_pkl) 

	

if __name__ == "__main__":
	main() 
 




# Load the data from the pickle file
with open("/storage/tmpuser/MetGen/result/canage.pkl",'rb') as file:
    object_file = pickle.load(file)

# Convert the data into a Pandas DataFrame
f = pd.DataFrame(object_file[0].items())

# Extract the dictionaries from the DataFrame
dicts = [f[1][i][0] for i in range(len(f))]

# Combine the dictionaries
combined_dict = {}
for d in dicts:
    combined_dict.update(d)

# Write the contents of the dictionary to a CSV file with headings
with open('/storage/tmpuser/MetGen/result/smiles_and_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['smiles', 'scores'])  # Write the headings to the first row of the file
    for key, value in combined_dict.items():
        writer.writerow([key, value])  # Write each key-value pair on a separate row

# Write the SMILES data to a CSV file
with open('/storage/tmpuser/MetGen/result/smiles.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['smiles'])  # Write the heading to the first row of the file
    for key in combined_dict.keys():
        writer.writerow([key])  # Write each SMILES on a separate row


###calculaion for average, novelty and diversity score for newly generated smiles

diversity = Evaluator(name = 'Diversity')
novelty = Evaluator(name = 'Novelty')

 
file = "/storage/tmpuser/MetGen/data/vocab_clean.txt"
with open(file, 'r') as fin:
	lines = fin.readlines()
train_smiles_lst = [line.strip().split()[0] for line in lines][:1000] 


## 5. run 
if __name__ == "__main__":

	# result_file = "result/denovo_from_" + start_smiles_lst[0] + "_generation_" + str(generations) + "_population_" + str(population_size) + ".pkl"
	# result_pkl = "result/ablation_dmg_topo_dmg_substr.pkl"
	# pkl_file = "result/denovo_qedlogpjnkgsk_start_ncncccn.pkl"
	pkl_file = "/storage/tmpuser/MetGen/result/canage.pkl"
	idx_2_smiles2f, trace_dict = pickle.load(open(pkl_file, 'rb'))
	# bestvalue, best_smiles = 0, ''
	topk = 100
	whole_smiles2f = dict()
	for idx, (smiles2f,current_set) in tqdm(idx_2_smiles2f.items()):
		whole_smiles2f.update(smiles2f)
		# for smiles,f in smiles2f.items():
		# 	if f > bestvalue:
		# 		bestvalue = f
		# 		print("best", f)
		# 		best_smiles = smiles 

	smiles_f_lst = [(smiles,f) for smiles,f in whole_smiles2f.items()]
	smiles_f_lst.sort(key=lambda x:x[1], reverse=True)
	best_smiles_lst = [smiles for smiles,f in smiles_f_lst[:topk]]
	best_f_lst = [f for smiles,f in smiles_f_lst[:topk]]
	avg, std = np.mean(best_f_lst), np.std(best_f_lst)
	print('average of top-'+str(topk), str(avg)[:5], str(std)[:5])
	#### evaluate novelty 
	t1 = time()
	nov = novelty(best_smiles_lst, train_smiles_lst)
	t2 = time()
	print("novelty", nov, "takes", str(int(t2-t1)), 'seconds')

	### evaluate diversity 
	t1 = time()
	div = diversity(best_smiles_lst)
	t2 = time()
	print("diversity", div, 'takes', str(int(t2-t1)), 'seconds')


	# ### evaluate mean of property 
	# for oracle_name in oracle_lst:
	# 	oracle = Oracle(name = oracle_name)
	# 	scores = oracle(best_smiles_lst)
	# 	avg = np.mean(scores)
	# 	std = np.std(scores)
	# 	print(oracle_name, str(avg)[:7], str(std)[:7])

	# for ii,smiles in enumerate(best_smiles_lst[:20]):
	# 	print(smiles, str(gsk(smiles)))
	# 	draw_smiles(smiles, "figure/best_"+oracle_name+"_"+str(ii)+'.png')





##Calculation of canage predictions and probabilities
FROM_Sign = '/storage/tmpuser/MetGen/Signaturizer/'
FROM_HOA = '/storage/tmpuser/MetGen/Hallmarks_of_aging/'

#parser = argparse.ArgumentParser()
#parser.add_argument('-i', help='path to smiles.csv, header name: smiles')
#parser.add_argument('-I', help='path to preprocessed_feature_file.pkl')
#parser.add_argument('-O', help='path to output directory')
#args = parser.parse_args()

# # Featurizing and Preprocessing Class

# In[10]:

print('Reading all classes.')

def Feature_Signaturizer(dat):
    print('Performing Signaturizer')
    sig_df=pd.DataFrame()
    sig_df['smiles']=dat['smiles'].tolist()
    desc=['A','B','C','D','E']
    for dsc in tqdm(desc):
        for i in range(1,6):
#            print(f"Processing descriptor {dsc}{i} for SMILES string: {dat['smiles'][i-1]}")
#             print('Performing '+dsc+str(i)+' Descriptor Calculation.')
            sign = Signaturizer(dsc+str(i))
            results = sign.predict(dat['smiles'].tolist())
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
    with open('/storage/tmpuser/MetGen/result/'+'preprocessed_feature_file.pkl', 'wb') as f:
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


# # HallMarks Of Aging Classes

# ## Altered Intercellular Communication

# In[62]:


class model_aic:
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
        model = joblib.load(FROM_HOA+'HOLY_AIC_model_svm_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Cellular Senescence

# In[63]:


class model_cs:
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
        model = joblib.load(FROM_HOA+'HOLY_CS_model_svm_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Deregulated Nutrient Sensing

# In[64]:


class model_dns:
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
        model = joblib.load(FROM_HOA+'HOLY_DNS_model_svm_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Epigenetic Alterations

# In[90]:


class model_ea:
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
        model = joblib.load(FROM_HOA+'HOLY_EA_model_svm_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Genomic Instability

# In[66]:


class model_gi:
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
        model = joblib.load(FROM_HOA+'HOLY_GI_model_RF_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Loss of Proteostasis

# In[67]:


class model_lp:
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
        model = joblib.load(FROM_HOA+'HOLY_LP_model_svm_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Mitochondrial Dysfunction

# In[68]:


class model_md:
    def __init__(self,test):        
        self.test = test
    def extract_feature(self,model,data):
        F_names = model.get_booster().feature_names
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
        model = joblib.load(FROM_HOA+'HOLY_MD_model_XGB_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Stem Cell Exhaustion

# In[69]:


class model_sce:
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
        model = joblib.load(FROM_HOA+'HOLY_SCE_model_et_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds


# ## Telomere Attrition

# In[70]:


class model_ta:
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
        model = joblib.load(FROM_HOA+'HOLY_TA_model_ET_HPTuned_fitted.pkl')
        test_filtered = self.extract_feature(model,test.drop(['smiles'],axis=1))
        probs = model.predict_proba(test_filtered)    
        preds = self.get_labels(probs)
        return probs,preds



# # Calculating Prediction Class

# In[99]:


def CanAge_Tox_Predictions(Sig_data):
    
    ####################################### Output dataframes #########################################################################    
    predictions = pd.DataFrame(columns=['smiles','Anti_Aging_Status','AIC_Prediction_Status','CS_Prediction_Status',
                                        'DNS_Prediction_Status','EA_Prediction_Status','GI_Prediction_Status','LP_Prediction_Status',
                                        'MD_Prediction_Status','SCE_Prediction_Status','TA_Prediction_Status'])
    
    probabilities = pd.DataFrame(columns=['smiles','Anti_Aging_Prob','AIC_Prob1','CS_Prob1','DNS_Prob1',
                                          'EA_Prob1','GI_Prob1','LP_Prob1','MD_Prob1','SCE_Prob1','TA_Prob1'])

    predictions['smiles'] = Sig_data['smiles']
    probabilities['smiles'] = Sig_data['smiles']
    
    ################################## Signaturizer Anti Aging Model #################################################################
    print('Onto Anti-Aging Predictions.')
    m0 = model_sign_aging(Sig_data)
    probs,preds = m0.test_model()
    probabilities['Anti_Aging_Prob'] = probs[:,1]
    predictions['Anti_Aging_Status'] = preds    
       
    ################################## Hallmarks of Aging Models ####################################################################
    print('Onto Hallmarks of aging predictions.')
    
    m1 = model_aic(Sig_data)
    m2 = model_cs(Sig_data)
    m3 = model_dns(Sig_data)
    m4 = model_ea(Sig_data)
    m5 = model_gi(Sig_data)   
    m6 = model_lp(Sig_data)
    m7 = model_md(Sig_data)
    m8 = model_sce(Sig_data)
    m9 = model_ta(Sig_data)
    
    probs,preds = m1.test_model()
    probabilities['AIC_Prob1'] = probs[:,1]
    predictions['AIC_Prediction_Status'] = preds    
    probs,preds = m2.test_model()
    probabilities['CS_Prob1'] = probs[:,1]
    predictions['CS_Prediction_Status'] = preds    
    probs,preds = m3.test_model()    
    probabilities['DNS_Prob1'] = probs[:,1]
    predictions['DNS_Prediction_Status'] = preds
    probs,preds = m4.test_model()    
    probabilities['EA_Prob1'] = probs[:,1]
    predictions['EA_Prediction_Status'] = preds 
    probs,preds = m5.test_model()    
    probabilities['GI_Prob1'] = probs[:,1]
    predictions['GI_Prediction_Status'] = preds      
    probs,preds = m6.test_model()
    probabilities['LP_Prob1'] = probs[:,1]
    predictions['LP_Prediction_Status'] = preds        
    probs,preds = m7.test_model() 
    probabilities['MD_Prob1'] = probs[:,1]
    predictions['MD_Prediction_Status'] = preds
    probs,preds = m8.test_model() 
    probabilities['SCE_Prob1'] = probs[:,1]
    predictions['SCE_Prediction_Status'] = preds
    probs,preds = m9.test_model() 
    probabilities['TA_Prob1'] = probs[:,1]
    predictions['TA_Prediction_Status'] = preds
    
    print('Done with CanAge-Hallmark Toxicity predictions.')
    
    with open('canage_probabilities.pkl', 'wb') as f:
        pickle.dump(probabilities, f)
        
    with open('canage_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    print('Saved CanAge predictions and prediction probabilities')
    return predictions

# In[43]:



def Predict_Meta(input_data):
    Sig_Input = Feature_Signaturizer(input_data)
    os.chdir('/storage/tmpuser/MetGen/result/')
    Preds_Out = CanAge_Tox_Predictions(Sig_Input)
    print('Total Counts : ' + str(len(input_data)) + ' Anti-Aging Hits : ' + str(len(Preds_Out[Preds_Out['Anti_Aging_Status']==1])))
    return

# # Testing Data

# In[ ]:


print('Reading input file -')


# In[ ]:


with open('/storage/tmpuser/MetGen/result/smiles.csv') as file:
    data = file.read().splitlines()


# In[ ]:


data = pd.DataFrame(data)
data = data.rename(columns=data.iloc[0]).drop(data.index[0])


print('Executing the script.')


# In[ ]:


Predict_Meta(data)

# read the pkl file into csv file
with open("/storage/tmpuser/MetGen/result/canage_probabilities.pkl", "rb") as f:
    object = pickle.load(f)
    
df = pd.DataFrame(object)
df.to_csv('canage_probabilities.csv')

fil= open("canage_probabilities.csv")
print(fil)
fil.close()

# read the pkl file into csv file
with open("/storage/tmpuser/MetGen/result/canage_predictions.pkl", "rb") as f:
    object = pickle.load(f)
    
df = pd.DataFrame(object)
df.to_csv('canage_predictions.csv')

fil= open("canage_predictions.csv")
print(fil)
fil.close()


###structure generation of the SMILES

# Read in the CSV file
df = pd.read_csv('/storage/tmpuser/MetGen/result/smiles_and_scores.csv')

# Extract the first SMILES string and its score
query_smiles = df.iloc[0]['smiles']
query_score = df.iloc[0]['scores']
print(f'Metformin: {query_smiles} (score: {query_score:.4f})')

# Sort the DataFrame by scores in descending order
df_sorted = df.sort_values(by='scores', ascending=False)

# Extract the top 10 SMILES strings with the highest scores using the first SMILES as the query
smiles_list = df_sorted.loc[df_sorted['smiles'] != query_smiles, 'smiles'].tolist()[:10]
smiles_list.insert(0, query_smiles)

# Printing Query and target molecules with their scores
target_mols = []
for i, smiles in enumerate(smiles_list[1:], start=1):
    print(f'Target {i}: {smiles} (score: {df_sorted[df_sorted["smiles"]==smiles]["scores"].values[0]:.4f})')

# Draw the structures of the top 10 SMILES strings with their corresponding names
target = []
legends = ['Query (Metformin)'] + [f'Target {i}' for i in range(1, 11)]
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    target.append(mol)

img = Draw.MolsToGridImage(target, molsPerRow=5, subImgSize=(300, 300), legends=legends, returnPNG=False)

# Save the image in PNG format
img.save('/storage/tmpuser/MetGen/result/metgen.png')
