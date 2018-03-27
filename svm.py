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
	"""Build the model using cvxopt quadratic
	programming optimizer.
	min(x) in 0.5(x^T)Px + q^Tx
	Where x is the lagrangian multiplier from the SVM equation
	P_ij = y_i*y_j*x_i*x_j^T
	Gx <= h
	Ax = b
	
	"""
	train = pd.read_csv('source_train.csv').as_matrix()
	x = matrix( train[ :, :2 ] )
	y = matrix( train[ :, 2 ] )
	C=1.0
	n = y.size[0]
	P = np.zeros((n,n))
	
	#build the P matrix
	for i in range(n):
		for j in range(n):
			P_ij = y[i]*y[j]*x[i,:] *x[j,:].T

			P[i, j] = P_ij[0]
	
	P=matrix(P)

	#q matrix
	q=matrix( [1 for i in range(n)], tc='d' )
	#b matrix
	b=matrix([0], tc='d')
	A=y.T

	h = matrix( np.zeros((2*n,1)), tc='d' )
	h[:n] = 0.0
	h[n:] = C
	G = matrix( np.zeros((2*n,n)), tc='d' )
	# G is basically to identity matrices stacked on top of eachother.
	G[:n, :] = -np.eye(n)
	G[n:, : ] = np.eye(n)
	try:
		sol = solvers.qp(P, q, G, h, A, b )
	except Exception as err:
		print(err)
		return P, q, G, h, A, b

	alphas = sol['x']
	Ws = np.zeros((1,2))
	for i in range(n):
		Ws = Ws + alphas[i], y[i]

	

	
	
	
