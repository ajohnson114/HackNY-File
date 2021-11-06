#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from copy import deepcopy
from pprint import pprint
import scipy
import math
from pprint import pprint

def step(W,H,S):
    """This is a direct implementation of the Multiplicative Update 
    step detailed in the paper but edited to our projects syntax
    
    Args:
        W = Dictionary matrix In paper: Z In our project: W
        H = Coding matrix In paper: W^t In our project: H
        S = Symmetric Matrix that is randomly defined same in both the paper and our project
    Input: Numpy arrays Output: Updated Numpy Arrays
    The dimensions should not change"""
    Ht = H.T
    W = W * ((X @ Ht)/((W @ H @ Ht)+10**-9))
    Ht = Ht * ((X.T @ W + (2 * M @ Ht @ S))/((Ht @ ((2 * S @ H @ Ht @ S)+ W.T @ W))+10**-9))
    S = S * (H @ M @ Ht)/(H @ Ht @ S @ H @ Ht)
    return W, Ht.T, S


#Function to be minimized
def loss_fn(X,W,H,M,S):
    """This is the function that needed to be optimized as per the paper.
    Input: numpy arrays Output: scalar loss value"""
    Ht = H.T
    loss1 = (np.linalg.norm((X - (W @ H)), 'fro')**2)
    loss2 = (np.linalg.norm((M - (Ht @ S @ H)),'fro')**2)
    loss = (1/2)*(loss1+loss2)
    return loss

def shifted_PPMI(documents, important_vocab, word_window = 0, neg_sample = 0):
    """This is supposed to caluclate the Shifted PPMI which 
    becomes the M used in Semantic NMF. It is based off the skip-gram
    model so neg_sample is the amount of negative sampling the researcher
    would use. For large corpora: 2-3 is ideal, for small: 5-20.

    Input: Preprocessed text in the form of documents (take out punctuation, etc) , important vocab (words in the X matrix),
    word window, and amount of negative sampling. Important vocab should be something that's iterable (list,tuple,etc)
    and the code is designed to be composed of the most important words in the TF-IDF perspective due to
    computation constraints

    Output: M as a numpy array of dim: important_vocab x important_vocab"""
    data = []
    Omega = []
    k = neg_sample
    word_window = word_window
    
    for document in documents:
        array = []
        for word in document.split():
            array.append(word)
        data.append(array)
    #Gets all the words without punctuation into a list 
    #for line in documents:
        #for word in line.split():
            #data.append(word)

    
    #Getting the amount of times the words appear within the word window
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(word_window):
                if j + k + 1  <= len(data[i][j]):
                    Omega.append((data[i][j],data[i][j+k]))
                else:
                    pass
                if j - k < 0:
                    pass
                else:
                    Omega.append((data[i][j],data[i][j-k]))
                
    #Counting total occurences of word pairings and individual words
    from collections import defaultdict
    Nij_counts = defaultdict(int)
    for (i,j) in Omega:
        Nij_counts[ (i,j) ] += 1
    #P(i) = Ni_counts/N and the same for j
    Ni_counts = defaultdict(int)
    Nj_counts = defaultdict(int)
    for (i,j), N_ij in Nij_counts.items():
        Ni_counts[ i ] += N_ij
        Nj_counts[ j ] += N_ij
    
    #Loading words into a data frame and dictionary
    # Now we are loading in the data from the dataframe (quick and kind of dirty)
    vocab = {}
    invvocab = important_vocab

    # count for words, not super efficient
    #for i in sorted(Ni_counts.items(), key=lambda kv: kv[1]): #Sorts and gives max to min for values a
    for i in important_vocab:
        vocab[i] = Ni_counts[i]
        
    V = len(invvocab)
    print("V = {}".format(V))
    # Given the vocabulary mapping above, we now fill in the Nij matrix.
    # NOTE - with large vocabs, this should be done using a *sparse* matrix!!!
    from scipy.sparse import csr_matrix
    Nij_np = csr_matrix((V, V))
    for m in range(V):
        i = invvocab[m]
        for n in range(V):
            j = invvocab[n]
            Nij_np[m, n] = Nij_counts[(i,j)]
    
    # marginalize to get the unigram counts
    Nij = tf.convert_to_tensor(Nij_np.todense(), dtype=tf.float32)
    N = tf.reduce_sum(Nij)
    Ni = tf.reduce_sum(Nij, axis=0)
    Nj = tf.reduce_sum(Nij, axis=1)

    PMI_np = tf.math.log((N * Nij) / (tf.einsum('i,j->ij', Ni, Nj)*k))
    PMI_np = tf.Session().run(PMI_np)
    shifted_PPMI = deepcopy(PMI_np)
                      
    for i in range(V):
        for j in range(V):
            if shifted_PPMI[i][j] < 0 or np.isnan(shifted_PPMI[i][j]):
                shifted_PPMI[i][j] = 0 
    

    return shifted_PPMI

    #Shifted PMI formula equivalent
    #log((N*Nij)/(NiNj*k))
    #N = Amount of words
    #Nij = Amount of cooccurrences
    #Ni (Nj) = Amount of occurrences of i (j)
    #k = Amount of negative samples
    
def Semantic_NMF(num_iterations,X,W,H,M,S):
    """Shows your current loss and allows you to specify how many times to update your matrices
    Input: All the matrices and the amount of times you want to update W,H, and S
    Output: Updated versions of W,H,S
    """
    for i in range(num_iterations):
        if i % 100 == 0:
            print("Your current loss is {:0.8f}".format(loss_fn(X,W,H,M,S)))
        W, H ,S = step(W,H,S)
    return W,H,S

def highest_PMI_words(amount):
    """ 
    Note that there was a slight bug when extracting corpus statistics,
    since the context window (w=5) is symmetric, PMI(i,j) should always equal
    PMI(j,i); however, due to improper handling of context during the first 5 words
    of a document, the statistics are ever-so-slightly distorted.
    
    Input: Amount of words you want printed with PMI values in descending order.
    Output: Prints word pairs with highest PMI values (neglects same word pairings ala (hello,hello)) 
    """
    printed = [] 
    def not_printed(term, context, printed):
        if (term,context) or (context,term) in printed:
            return False
        return True

    
    # look at the biggest PMIs
    sorted_ind = np.argsort(M, axis=None)

    print('--- Word pairs with the notable PMIs ---\n')
    for i in range(1,amount):
        ind = np.unravel_index(sorted_ind[-i], M.shape)
        term, context = important_vocab[ind[0]], important_vocab[ind[1]]
        if term != context and not_printed(term, context, printed):
            printed.append((term,context))
            print('{:8} {:14} (PMI = {:0.4f})'.format(term, context, float(M[ind])))

