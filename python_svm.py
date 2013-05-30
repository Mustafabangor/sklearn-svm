import pickle
import math
import pymongo
import numpy as np
import scipy
import sklearn
from sklearn import svm
from data_processor import TrainingSet, DataSet
from interactive_spellcheck import InteractiveSpellchecker

class ScikitSvmPipeline:
    def __init__(self,complete_datastore,vocabstores):
        self.trainset = TrainingSet(complete_datastore)
        self.testset = DataSet(complete_datastore)
        self.datastore = self.trainset.storage
        self.spellchecker = InteractiveSpellchecker()
        self.vocabstores = vocabstores
        # a list of ten sparse matrices, each of which is used to train a different classifier during crossval
        self.training_data_sets = []
        self.crossval_data_sets = []
        # the ten classifiers that are trained during cross validation
        self.classifiers = []
        self.crossval_stats = []

    def set_train_data(self):
        self.pos_training_ids, self.neg_training_ids = self.trainset.choose()
        for id in self.pos_training_ids + self.neg_training_ids:
            doc = self.datastore.find_one({'id':id})
            mongoid = doc['_id']
            self.datastore.update({'_id':mongoid},{'$set':{'train':True}})
        self.num_pos_training = len(self.pos_training_ids)
        self.num_neg_training = len(self.neg_training_ids)

    def cross_val(self):
        p = (self.num_pos_training)/10
        n = (self.num_neg_training)/10
        i = 1 
        while (i<=10):
            self.datastore.update({'id':{'$exists':True}},{'$set':{'tokens':None}},multi=True) 
            current_max_index = max(self.vocabstores[i-1].distinct('index')+[0])
            self.vocab_index_tracker = current_max_index+1 if (current_max_index>0) else current_max_index 
            leave_out_pos = self.pos_training_ids[(i-1)*p:i*p]
            leave_out_neg = self.neg_training_ids[(i-1)*n:i*n]
            leave_in_pos = self.pos_training_ids[:(i-1)*p] + self.pos_training_ids[i*p:]
            leave_in_neg = self.neg_training_ids[:(i-1)*n] + self.neg_training_ids[i*n:] 
            self.process_data(i-1,leave_in_pos + leave_in_neg,True)
            self.get_idfs(self.vocabstores[i-1],(leave_in_pos+leave_in_neg))
            self.training_data_sets.append(self.trainset.get_labelled_data(self.vocabstores[i-1],leave_in_pos,leave_in_neg))
            self.process_data(i-1,leave_out_pos + leave_out_neg,False)
            self.crossval_data_sets.append(self.trainset.get_labelled_data(self.vocabstores[i-1],leave_out_pos,leave_out_neg))
            current_classifier = self.get_svm(self.training_data_sets[i-1][0],self.training_data_sets[i-1][1])
            self.classifiers.append(current_classifier)
            prec, rec = self.eval(self.crossval_data_sets[i-1][0],current_classifier,len(leave_out_pos)) 
            self.datastore.save({'crossval_run':i,'classifier':pickle.dumps(current_classifier),'recall':rec,'precision':prec})
            print "JUST FINISHED CROSSVAL ITER %d \n\n" % i
            i += 1
            
    def get_idfs(self,vocabstore,ids):
        for id in ids:
            doc = self.datastore.find_one({'id':id})
            for token in doc['tokens']:
                vocab_data = vocabstore.find_one({'term':token})
                expression = (len(ids)/(vocab_data['df']))
                mongoid = vocab_data['_id']
                vocabstore.update({'_id':mongoid},{'$set':{'idf':math.log(expression,10)}})
 
    """
    Tokenizes every document in the data set.
    """
    def process_data(self,crossval_iter,ids,is_leavein):
        for id in ids:
            doc = self.datastore.find_one({'id':id}) 
            mongoid = doc['_id']
            correct = self.spellchecker.process_text(doc['orig'])
            self.datastore.update({'_id':mongoid},{'$set':{'tokens':correct}})
            if is_leavein: 
                self.update_vocab(self.vocabstores[crossval_iter],correct)

    def update_vocab(self,vocabstore,tokens):
        try:
            for token in set(tokens):
                if token not in vocabstore.distinct('term'):
                    vocabstore.save({'term':token,'df':1,'index':self.vocab_index_tracker,'idf':None})
                    self.vocab_index_tracker += 1
                else:
                    doc = vocabstore.find_one({'term':token})
                    mongoid = doc['_id']
                    vocabstore.update({'_id':mongoid},{"$inc":{'df':1}}) 
        except TypeError: pass

    def get_svm(self,X,y): 
        classifier = svm.SVC()
        return classifier.fit(X,y)
    
    """
    accepts: the data set to classify; an SVM classifier trained during one of the cross validation folds;
    the number of positive labels for ground truth; the number of negative labels for ground truth
    returns: precision; recall
    """
    def eval(self,data,classifier,num_pos):
        hypotheses = classifier.predict(data) 
        """
        precision:
        (number elements correctly labelled positive)/(number elements labelled positive)
        recall:
        (number elements correctly labelled positive)/number positive elements)
        return (precision,recall)
        """
        bools = np.logical_and(hypotheses[:num_pos],np.ones(num_pos,dtype=int))
        num_truepos = (bools[bools==True]).size
        precision = float(num_truepos)/sum(hypotheses)
        recall = float(num_truepos)/num_pos
        return (precision,recall)

    def select_model(self):
        print 'prec,rec'
        i = 1
        while (i <= 10):
            for doc in self.datastore.find({'crossval_run':i}):
                print doc['precision'],doc['recall']
                i += 1
