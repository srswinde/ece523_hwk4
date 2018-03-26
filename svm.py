import pandas as pd
from sklearn import svm
import pylab
import numpy as np
import cvxopt 
from cvxopt import matrix
from cvxopt import solvers

def mesh(samples, step=0.2):
    Min = samples[:,0].min(), samples[:,1].min()
    Max = samples[:,0].max(), samples[:,1].max()
    xx, yy = np.meshgrid(np.arange(Min[0], Max[0], step ), np.arange(Min[1], Max[1], step ))
    return xx, yy


def get_train():
	train = pd.read_csv('source_train.csv').as_matrix()

	return train[ :, :2 ], train[ :, 2 ]


def sklearn_Ws():
	train = pd.read_csv('source_train.csv').as_matrix()

	x = train[ :, :2 ]
	y = train[ :, 2 ]

	clf = svm.LinearSVC()
	clf.fit(x,y)

	xx, yy = mesh(x)
	z=clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

	pylab.contourf(xx, yy, z, cmap=pylab.cm.coolwarm)
	pylab.plot(x[:, 0][y==1], x[:,1][y==1], 'k.')
	pylab.plot(x[:, 0][y==-1], x[:,1][y==-1], 'kx')

	pylab.show()




def cvxopt_Ws():
	train = pd.read_csv('source_train.csv').as_matrix()
	x = matrix( train[ :, :2 ] )
	y = matrix( train[ :, 2 ] )
	C=1.0
	n = y.size[0]
	P = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			P_ij = y[i]*y[j]*x[i,:] *x[j,:].T

			P[i, j] = P_ij[0]

	P=matrix(P)
	q=matrix( [1 for i in range(n)], tc='d' )
	b=matrix([0 for i in range(n)])
	A=y.T
	h = matrix( np.array( [[0], [C]] ), tc='d' ) # THe bounds of lagrange multiplier
	G = matrix( np.array( [[-1 for i in range(n)], [1 for i in range(n)]] ), tc='d' )
	sol = solvers.qp(P, q, G, h, A, b )
	

	
	
	
