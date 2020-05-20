# -*- coding: utf-8 -*-
"""
@author: luk10

 Unit testing for the data models
"""

import numpy as np
import matplotlib.pyplot as plt

import datamodels as dm
        
features = ['daily_actions', 'daily_sessions', 'tenure', 'days_played', 'max_level']

def runmodels():
    zr = dm.ZeroR('train.csv', [], 'churn_yn')
    zr.train()
    zr_train = zr.test('train.csv')
    zr_test  = zr.test('test.csv')
    
    dt = dm.DecisionTree('train.csv', features, 'churn_yn')
    dt.train()
    dt_train = dt.test('train.csv')
    dt_test  = dt.test('test.csv')
    
    rf = dm.RandomForest('train.csv', features, 'churn_yn')
    rf.train()
    rf_train = rf.test('train.csv')
    rf_test  = rf.test('test.csv')
    
    lr = dm.LogisticRegression('train.csv', 
                        ['daily_sessions', 'days_played', 'max_level'],
                        'churn_yn')
    lr.train()
    lr_train = lr.test('train.csv')
    lr_test  = lr.test('test.csv')
    
    nn = dm.NeuralNetwork('train.csv', features, 'churn_yn')
    nn.train()
    nn_train = nn.test('train.csv')
    nn_test  = nn.test('test.csv')
    
    return {'ZeroR': (zr_train, zr_test),
            'DecTree': (dt_train, dt_test),
            'RandForest': (rf_train, rf_test),
            'LogRegression': (lr_train, lr_test),
            'ANN': (nn_train, nn_test)}