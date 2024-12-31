#This code defines a class called "DPPModel" that implements a Determinantal Point Process (DPP) algorithm. The DPP algorithm is used to select a subset of items from a larger set, based on their similarity to each other. The class takes as input a list of smiles strings (smiles_lst), a similarity matrix (sim_matrix), scores for each item (f_scores), the number of items to select (top_k), and a lambda parameter (lamb). It initializes the class by defining the input variables as attributes, creating a kernel matrix from the similarity matrix and scores, and calculating the log determinant of the kernel matrix (log_det_S) and the sum of the scores (log_det_V). The class also has a method called "dpp" which runs the DPP algorithm. It initializes empty arrays for the selected items (Yg) and the coefficients (c) and the diagonal of the kernel matrix (d). It then runs a while loop to select items based on the highest score in the d array, until the desired number of items (top_k) is reached. The selected items are returned as a list of smiles strings. At the end of the code, the DPPModel class is instantiated with some random values for the input variables, and the dpp method is called. The returned list of selected items is printed to the console. However, there is a mistake, the parameter top_k is missing in the instantiation of the class.





import numpy as np
import math
np.random.seed(1)

class DPPModel(object):
    def __init__(self, smiles_lst, sim_matrix, f_scores, top_k, lamb):
        self.smiles_lst = smiles_lst 
        self.sim_matrix = sim_matrix # (n,n)
        self.lamb = lamb
        self.f_scores = np.exp(f_scores) * self.lamb # (n,) 
        self.max_iter = top_k 
        self.n = len(smiles_lst)
        self.kernel_matrix = self.f_scores.reshape((self.n, 1)) \
                             * sim_matrix * self.f_scores.reshape((1, self.n))
        self.log_det_V = np.sum(f_scores) * self.lamb 
        self.log_det_S = np.log(np.linalg.det(np.mat(self.kernel_matrix)))

    def dpp(self): 
        c = np.zeros((self.max_iter, self.n))
        d = np.copy(np.diag(self.kernel_matrix))  ### diagonal
        j = np.argmax(d)
        Yg = [j]
        _iter = 0
        Z = list(range(self.n))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                if _iter == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:_iter, j], c[:_iter, i])) / np.sqrt(d[j])
                c[_iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)
            Yg.append(j)
            _iter += 1

        return [self.smiles_lst[i] for i in Yg], self.log_det_V, self.log_det_S 



if __name__ == "__main__":
    rank_score = np.random.random(size=(100)) 
    item_embedding = np.random.randn(100, 5) 
    item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
    sim_matrix = np.dot(item_embedding, item_embedding.T) 

    dpp = DPPModel(smiles_lst=list(range(100)), sim_matrix = sim_matrix, f_scores = rank_score, top_k = 10)
    Yg = dpp.dpp() 
    print(Yg)




