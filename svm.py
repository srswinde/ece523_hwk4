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


def get_data(fname):
	train = pd.read_csv(fname).as_matrix()

	return matrix( train[ :, :2 ] ), matrix( train[ :, 2 ] )


def svm_sklearn():
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
	return clf




def svm_cvxopt(x, y, C=1.0, B=0, Ws=None):
	"""Build the model using cvxopt quadratic
	programming optimizer.
	min(x) in 0.5(x^T)Px + q^Tx
	Where x is the lagrangian multiplier from the SVM equation
	P_ij = y_i*y_j*x_i*x_j^T
	Gx <= h
	Ax = b
	
	"""


	if Ws == None and B != 0:
			raise ValueError("Ws must have a value matrix(2x1) if B is not 0")
			
	n = y.size[0]
	P = np.zeros((n,n))
	
	#build the P matrix
	for i in range(n):
		for j in range(n):
			P_ij = y[i]*y[j]*x[i,:] *x[j,:].T

			P[i, j] = P_ij[0]
	
	P=matrix( P )

	#q matrix
	q=matrix( [1 for i in range(n)], tc='d' )

	#b matrix
	b=matrix([0], tc='d')
	A=y.T

	#h matrix 
	h = matrix( np.zeros((2*n,1)), tc='d' )
	h[:n] = 0.0
	h[n:] = C
	
	#G matrix
	G = matrix( np.zeros((2*n,n)), tc='d' )
	# G is basically two identity matrices stacked on top of eachother.
	G[:n, :] = -np.eye(n)
	G[n:, : ] = np.eye(n)


	try:
		sol = solvers.qp( P, q, G, h, A, b )
	except Exception as err:
		print(err)
		return P, q, G, h, A, b

	alphas = sol['x']
	if Ws:
		
		W = matrix( np.zeros((1,2)) ) + B*Ws
	else:
		W = matrix( np.zeros((1,2)) ) 
	
	try:
		for i in range(n):
			W = W + alphas[i]*y[i]*x[i, :]
	except Exception as err:
		print( err )
		return alphas, y, x

	return W

	

	
	
def main():
	x_source, y_source = get_data('source_train.csv')
	x_target, y_target = get_data('target_train.csv')

	W_s = svm_cvxopt( x_source, y_source )
	W_t = svm_cvxopt( x_target, y_target, B=1e-1, Ws=W_s )
	
	print (W_s)
	print (W_t)
	
main()
