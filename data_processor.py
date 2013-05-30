"""
dict keys: number of hours since 10/28 noon
values:
precipitation intensity in that hour
number of tweets in that hour
"""
from darksky_twitter_req import Tweet
import pickle
import glob
import enchant
import enchant.checker
from enchant.checker.CmdLineChecker import CmdLineChecker
from interactive_spellcheck import InteractiveSpellchecker
import pymongo
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import random
import math

class DataSet(object):
    def __init__(self,collection):
        self.storage = collection
    
    """
    Returns a compressed sparse row matrix that contains only unlabelled test data.
    """
    def get_unlabeled_data(self):
        data_points = []
        for doc in self.storage.find({'train':False,'label':None}):
            data_points.append(doc['vec'])
        matrix = np.asmatrix(data_points)
        return csr_matrix(matrix)
    
    def get_pos_neg_ids(self,vocabstore):
        pos = []
        neg = [] 
        for doc in self.storage.find({'train':False,'tokens':{'$exists':True}}): 
            label = doc['label']
            if (label==1): pos.append(doc['id'])
            elif (label==0): neg.append(doc['id'])
            else:
                print "The following doc has no label"
                print doc
        return (pos,neg)

    """
    Returns a compressed sparse row matrix that contains only labeled test data 
    for calculating precision and recall of the classification hypotheses.
    """
    def get_labelled_data(self,vocabstore,pos,neg):
        y_1 = np.ones(len(pos))
        y_0 = np.zeros(len(neg))
        y = np.concatenate((y_1,y_0),axis=0)
        record = 0
        rows = []
        cols = []
        data = []
        vocabsize = vocabstore.count()
        for id in (pos + neg):
            doc = self.storage.find_one({'id':id})
            tokens = doc['tokens']
            for token in tokens:
                freq = tokens.count(token)
                col,val = self.get_tfidf_data(vocabstore,token,freq)
                if col:
                    rows.append(record)
                    cols.append(col)
                    data.append(val)
            record += 1    
        X = coo_matrix((data,(rows,cols)),shape=(record,vocabsize))
        return (csr_matrix(X),y)
       
    def get_tfidf_data(self,vocabstore,token,freq):
        try:
            term_data = vocabstore.find_one({'term':token})
            # (col,data)
            return (term_data['index'],freq*term_data['idf'])
        except TypeError:
            return (None,None)
    
    """
    Returns array of labels for the labelled test examples in the test set.
    """
    def get_labels(self):
        pos = np.ones(self.num_pos_labels)
        neg = np.zeros(self.num_neg_labels)
        return np.concatenate((pos,neg),axis=0)

class TrainingSet(DataSet):
    """
    Returns a list of ids of data points for the training set that we will perform
    ten-fold cross-validation on.
    """
    def choose(self):
        pos_ids = [doc['id'] for doc in self.storage.find({'label':1})]
        random.shuffle(pos_ids)
        """
        Want >= 90% of the positive labeled examples to be in the training set.
        Make the number of positive labels in training set divisible by 10 to ensure
        we can perform ten-fold cross validation.
        """
        self.crossval_pos_size = int(math.ceil(len(pos_ids)*.09))
        self.num_pos_training = 10 * self.crossval_pos_size
        pos_train_ids = pos_ids[:self.num_pos_training]
        neg_ids = [doc['id'] for doc in self.storage.find({'label':0})]
        random.shuffle(neg_ids)
        self.crossval_neg_size = int(math.ceil(len(neg_ids)*.09))
        self.num_neg_training = 10 * self.crossval_neg_size
        neg_train_ids = neg_ids[:self.num_neg_training]
        return (pos_train_ids,neg_train_ids)
