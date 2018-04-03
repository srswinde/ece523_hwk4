from sklearn.ensemble  import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

import pandas as pd
import pylab
import numpy as np
import re

def bagging( x, y,  n_estimators ):
	bagger = BaggingClassifier( DecisionTreeClassifier(max_depth=20, min_samples_leaf=1), n_estimators=n_estimators, max_samples=0.5, max_features=0.5)
	return bagger

	
def boosting( x, y, n_estimators ):
	booster = AdaBoostClassifier( n_estimators=n_estimators, learning_rate=0.1 )
	booster.fit( x, y)
	return booster

class TrainData:
	"""A convenient way to grab data without saving them to disk."""
	urls = (
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/thyroid_train.csv", 
		"https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/echocardiogram_train.csv", 
		"https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/breast-cancer_train.csv", 
		"https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/congressional-voting_train.csv", 
		"https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/conn-bench-sonar-mines-rocks_train.csv", 
		"https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/cylinder-bands_train.csv", 
		"https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/echocardiogram_train.csv", 
		"https://github.com/gditzler/UA-ECE-523-Sp2018/raw/master/data/haberman-survival_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/hayes-roth_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/heart-hungarian_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/hill-valley_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/horse-colic_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/ionosphere_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/image-segmentation_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/mammographic_train.csv", 
		"https://raw.githubusercontent.com/gditzler/UA-ECE-523-Sp2018/master/data/monks-1_train.csv", 
	)
	cache =  None
	def __getitem__( self, key ):
		if self.cache is None:
			self.cache = {}
			
		if key in self.cache:
			data = self.cache[key][:]

		else:
			data = pd.read_csv( self.urls[key], header=None ).as_matrix()
			self.cache[key] = data[:]
			

		return data

	def __iter__(self):
		for ii in range( len( self.urls ) ):
			yield self.__getitem__(ii)


	def __call__( self ):
		"""Cache the data"""
		for data in self.__iter__():
			pass
		
	def xy( self, key ):
		data = self.__getitem__( key )
		x, y = data[:, :-1], data[:, -1]
		return x, y

	def __len__(self):
		return len(self.urls)


class TestData( TrainData ):
	"In case we want the test data"
	def __init__( self ):
		self.urls = [ url.replace("_train", "_test") for url in self.urls ]



	
	

def classify( traindata=None, testdata=None, classifier_name='bagging' ):

	"""Most of the work is done here...

		This funciton Fits ensemble classifiers for 15 different data sets
		defined in traindata. The number of classifiers
		is set by range_estimators (2 to 50). I Use cross_val_score
		to score the data and then average the score for each
		n_estimators. You should end up with a dataframe
		indexed by n_estimators with one column--'err'. 

		Enjoy!
	"""

	if traindata is None:
		traindata = TrainData()
	if testdata is None:
		testdata = TestData()

	#we only want to do one of two things
	assert( classifier_name  in ['bagging', 'boosting'] )
	
	
	if classifier_name == 'bagging':
		clf_funct = bagging
	else:
		clf_funct = boosting
	
	#initialize the output dataframe
	output = pd.DataFrame(columns=('dataset', 'n_estimators', 'err') )
	
	#how many estimators
	range_estimators = range( 2, 50)
	counter = 0

	
	for ii in range( len( traindata ) ):
	#iterate through 15 datasets

		for n_estimators in range_estimators:
		#iterated throught n_estimators

			x, y = traindata.xy( ii )
			#setup the classifier
			clf = clf_funct( x, y, n_estimators )
			
			#score it. 
			scores = cross_val_score( clf, x, y  )
			#grab the err
			err = 1-scores.mean()
			
			#populate the output dataframe
			output.loc[counter] = ( traindata.urls[ii].split('/')[-1], n_estimators, err )
			counter+=1

	#Too much to plot each dataset. Lets average there err together. 
	avg_output = pd.DataFrame(columns=( 'err', ), index=range_estimators )
	for n_estimators in range_estimators:
		avg_output.loc[n_estimators] = output[output.n_estimators==n_estimators].err.mean()

	return avg_output
	

	
	
def main(df=None):
	""" Call classifier and plot the result or if df is not none
		plot the df. """

	#train and score the ensemble. 
	if df == None:
		df=pd.DataFrame( columns=('boosting', 'bagging') )
	
		for method in ("bagging", 'boosting'):
			df[method]=classify(classifier_name=method).err
	else:
		df=pd.read_pickle(df)


	#plot the data. 
	fig=pylab.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Boosting Vs Bagging")
	ax.set_xlabel("Number of Estimators")
	ax.set_ylabel("Average error of All Datasets")
	ax.plot(df.boosting, label='Boosting')
	ax.plot(df.bagging, label='Bagging')
	
	ax.legend()
	pylab.savefig("ensembles.png")
	df.to_pickle('ensembles.pkl')
	return df	
	
		
	

