#This is a class for creating a PyTorch dataset from a list of SMILES strings. The class inherits from torch.utils.data.Dataset and overrides the __init__, __len__, and __getitem__ methods.

#The __init__ method initializes the class with a list of SMILES strings.

#The __len__ method returns the number of SMILES strings in the list.

#The __getitem__ method returns the SMILES string at a specific index in the list.

#This class can be used to create a PyTorch dataloader for training or evaluating a model on a dataset of SMILES strings.




import torch 



class Molecule_Dataset(torch.utils.data.Dataset):
	def __init__(self, smiles_lst):
		self.smiles_lst = smiles_lst

	def __len__(self):
		return len(self.smiles_lst)

	def __getitem__(self, idx):
		return self.smiles_lst[idx]




