# -*- coding: utf-8 -*-
"""
@author luk10

 Builds the data models
"""

from sklearn import linear_model, tree, metrics, ensemble, model_selection, \
    neural_network, preprocessing
from dataset import Dataset

class ModelBase:
    def __init__(self, datafile, features, target):
        self.data     = Dataset(datafile)
        self.features = features
        self.target   = target

    def predict(self, X):
        return
    
    # Evaluates the trained model on a given dataset
    def test(self, testfile):
        testdata = Dataset(testfile).df
        
        X = self.predict( testdata[self.features] )
        y = testdata[self.target]
        
        return metrics.classification_report( y, X )

class LogisticRegression(ModelBase):
    def __init__(self, datafile, features, target):
        super().__init__(datafile, features, target)
        
        self.model = linear_model.LogisticRegression(solver='lbfgs')
    
    def train(self):
        X = self.data.df[self.features]
        y = self.data.df[self.target]
                
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class DecisionTree(ModelBase):
    def __init__(self, datafile, features, target):
        super().__init__(datafile, features, target)
        
        self.model = tree.DecisionTreeClassifier(max_depth=3)

    def train(self):
        X = self.data.df[self.features]
        y = self.data.df[self.target]
                
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)


class RandomForest(ModelBase):
    def __init__(self, datafile, features, target):
        super().__init__(datafile, features, target)
        
        self.model = ensemble.RandomForestClassifier(max_depth=3, n_estimators=50)

    def train(self):
        X = self.data.df[self.features]
        y = self.data.df[self.target]
        
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    

class NeuralNetwork(ModelBase):
    def __init__(self, datafile, features, target):
        super().__init__(datafile, features, target)
        
        self.scaler = preprocessing.StandardScaler()
        
        self.model = neural_network.MLPClassifier(alpha=1e-5,
                                                  hidden_layer_sizes=(200,500,200,),
                                                  max_iter=2000)

    def train(self):
        X = self.data.df[self.features]
        y = self.data.df[self.target]
        

        self.scaler.fit(X)
        X = self.scaler.transform(X)
        
        self.model.fit(X, y)
        
    def predict(self, X):
        X = self.scaler.transform(X)
        
        return self.model.predict(X)


# Baseline accuracy model, always predicts the most common target value
class ZeroR(ModelBase):
    def __init__(self, datafile, features, target):
        super().__init__(datafile, features, target)
        
        self.predictval = 0
        
    def train(self):
        positives = self.data.df[self.target].sum()
        setsize = len(self.data.df)
        
        if positives > setsize/2:
            self.predictval = 1
    
    def predict(self, X):
        return [self.predictval for _ in X.iterrows()]

    