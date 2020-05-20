# -*- coding: utf-8 -*-
"""
@author: luk10

 Handles creation and access of modified dataset
"""

import pandas as pd
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

ROOTDIR   = './bnsdataset/'
LABELFILE = ROOTDIR + 'test_labels.csv'
TRAINDIR  = ROOTDIR + 'test1/'

VERBOSE = False

class Preprocess:
    def __init__(self):
        self.churndata = pd.read_csv(LABELFILE,
                                     usecols=['actor_account_id', 'churn_yn'],
                                     index_col='actor_account_id')
        self.df = pd.DataFrame()
    
        self.read()
    
    def getcsvlist(self):
        filelist = listdir(TRAINDIR)
        return filelist
    
    def getaccountchurn(self, accid):
        return self.churndata.loc[accid]['churn_yn']
    
    def parsecsv(self, filename):
        a_accid      = filename.split('.')[0]
        a_actions    = 0
        a_sessions   = 0
        a_daysplayed = 1
        a_maxlvl     = 0
        a_tenure     = pd.Timedelta(0)
        a_churn      = self.getaccountchurn(a_accid)
        
        data = pd.read_csv(TRAINDIR+filename,
                           usecols=['time', 'session', 'actor_level'],
                           parse_dates=['time'])
        
        data.query('session != 0', inplace=True) # Ignore entries without session ids

        curdate    = data.iloc[0]['time'].floor('D')
        lastsesh   = 0
        for _, act in data.iterrows():
            if act['session'] != lastsesh:
                a_sessions += 1
                lastsesh = act['session']
                
            if act['actor_level'] > a_maxlvl:
                a_maxlvl = act['actor_level']
                
            # On new day
            if act['time'] - curdate > pd.Timedelta('1D'):
                curdate = act['time'].floor('D')
                a_daysplayed += 1
        
        # Average out by day
        a_actions  = len(data) / a_daysplayed
        a_sessions /= a_daysplayed
        
        tenure = data['time'].iloc[-1] - data['time'].iloc[0]
        a_tenure = tenure.total_seconds()/3600 # Convert timedelta to hours float
        
        return pd.Series({'account_id':a_accid, 'daily_actions':a_actions,
                          'daily_sessions':a_sessions, 'tenure':a_tenure,
                          'days_played':a_daysplayed, 'max_level':a_maxlvl,
                          'churn_yn':a_churn})
    
    def read(self):
        for csv in self.getcsvlist():
            print('Reading file', csv)
            self.df = self.df.append( self.parsecsv(csv), ignore_index=True )
    
    def save(self, filename):
        print('Saving to file', filename)
        self.df.to_csv(filename, index=False)


class Dataset:
    def __init__(self, filename):
        self.df = self.read(filename)

    def read(self, filename):
        return pd.read_csv(filename)
    
    def histo(self, feat, target):
        d = self.df
        d[d[target]==0][feat].hist( bins=30, range=(0,d[feat].max()),
                                        alpha=0.5, label='n' )
        d[d[target]==1][feat].hist( bins=30, range=(0,d[feat].max()),
                                        alpha=0.5, label='y' )
        plt.xlabel(feat)
        plt.legend(loc='upper right')
        plt.grid(b=None)
        plt.show()

    def scatter(self, featx, featy, target):
        self.df[target] = pd.to_numeric(self.df[target], downcast='integer')
        plt.scatter( self.df[featx], self.df[featy],
            color=[ ('red','blue')[i] for i in self.df[target] ], alpha=0.5)
        