import random
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

INFO = False

class DataSet():
	def __init__(self, x, y):
		self.X = x
		self.y = y


def load_clean_data():
	# This can be used to load the dataset
	data = pd.read_csv("./adult.csv", header=1, na_values='?')

	if INFO:
		print( data.shape )

	data.dropna(inplace=True)
	data = data.to_numpy()

	x = data[:,:-1]
	y = data[:,-1].reshape(-1,1)
	return x, y


def preprocess_data(x, y):
	categorical = [ ]
	for i in range( x.shape[1] ):
		if isinstance( x[0,i], str ):
			categorical.append( i )

	if INFO:
		for i in categorical:
			values = [ ]
			for value in x[:,i]:
				if not value in values:
					values.append(value)
			print(values)

	transformers = [
		( 'scaler', StandardScaler(), [ 0, 2, 4, 10, 11, 12 ] ),
		( 'onehot', OneHotEncoder(), [ 1, 3, 5, 6, 7, 8, 9, 13 ] ),
		#( 'ordinal', OrdinalEncoder(), [ 1, 3, 5, 6, 7, 8, 9, 13 ] ),
	]

	ct = ColumnTransformer( transformers=transformers, remainder='passthrough' )
	x = ct.fit_transform( x ).toarray()
	#x = ct.fit_transform( x )

	if INFO:
		print( f"Pre-PCA: {x.shape}" )

	labelencoder = LabelEncoder()
	y = labelencoder.fit_transform( y.ravel() )

	if INFO:
		print( y.shape )
		print( f"Classes: {onehot.categories_}" )

	train_x, test_x, train_y, test_y = train_test_split( x, y, test_size=0.33 )

	pca = PCA( n_components=0.95, svd_solver='full' )
	train_x = pca.fit_transform(train_x)
	test_x = pca.transform(test_x)

	train = DataSet( train_x, train_y )
	test = DataSet( test_x, test_y )
	return train, test

def train_model( model, train ):
	model.fit( train.X, train.y )

	conf_mat = evaluate_model( model, train )
	return model, conf_mat


def evaluate_model( model, test ):
	y_pred = model.predict( test.X )

	conf_mat = confusion_matrix( test.y, y_pred )
	print(conf_mat)
	return conf_mat


def add_errors( y, fraction=0.5 ):
	# Flip fraction*len(data) of the labels in copy
	idx = np.arange( y.shape[0] )
	np.random.shuffle(idx)

	mx_idx = round(fraction*y.shape[0])
	if INFO:
		print( mx_idx )
	idx = idx[:mx_idx]
	y[idx] = y[idx] ^ 1

	return y

def perturb( y, mn=10, mx=100 ):
	mn = random.randrange(mn)
	mx = random.randrange(mx)
	if mn > mx:
		tmp = mx
		mx = mn
		mn = tmp
	elif mn == mx:
		mx += 1
	
	print( f"New range {mn} - {mx}" )
	y = minmax_scale( y, feature_range=( mn, mx ) )
	return y

def train_eval( train, test, classifiers ):
	for classifier in classifiers:
		print( f"Training: {classifier}" )
		classifier = classifiers[classifier]
		classifier['model'], classifier['train_conf'] = train_model( classifier['model'], train )
		classifier['test_conf'] = evaluate_model( classifier['model'], test )
		print("")

def visualize( dt, model ):
	h = 0.02  # step size in the mesh
	x_min, x_max = dt.X[:, 0].min() - 1, dt.X[:, 0].max() + 1
	y_min, y_max = dt.X[:, 1].min() - 1, dt.X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Plot decision boundary of Linear SVM
	Z_linear = model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z_linear = Z_linear.reshape(xx.shape)
	plt.contourf( xx, yy, Z_linear, cmap=plt.cm.Paired, alpha=0.8)
	plt.scatter( dt.X[:, 0], dt.X[:, 1], c=dt.y, cmap=plt.cm.Paired)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()


if __name__ == '__main__':
	x, y = load_clean_data()
	train, test = preprocess_data( x, y )

	classifiers = {
		'svc': { 'model': SVC(gamma='auto'), 'train_conf': None, 'test_conf': None },
		'knn': { 'model': KNeighborsClassifier(n_neighbors=10), 'train_conf': None, 'test_conf': None },
		'tree': { 'model': DecisionTreeClassifier(), 'train_conf': None, 'test_conf': None },
		'qda': { 'model': QuadraticDiscriminantAnalysis(), 'train_conf': None, 'test_conf': None }
	}
	
	#Undo One-Hot for sklearn
	#train.y = np.argmax(train.y, axis=1)
	#test.y = np.argmax(test.y, axis=1)

	print( f"Training on NORMAL dataset" )
	print( f"--------------------------------------" )
	train_eval( train, test, classifiers )
	#visualize( train, classifiers['svc']['model'] )
	
	print( f"\n\nTraining on FLIPPED datasets" )
	print( f"--------------------------------------" )
	for fraction in range( 1, 7, 2 ):
		print( f"Fraction: {fraction/10}" )
		error_train = DataSet( train.X, add_errors( train.y, fraction=fraction/10 ) )
		train_eval( error_train, test, classifiers )
	
	print( f"\n\nTraining on PERTURBED datasets" )
	print( f"--------------------------------------" )
	for _ in range( 5 ):
		error_train = DataSet( train.X, perturb( train.y ) )
		train_eval( error_train, test, classifiers )
