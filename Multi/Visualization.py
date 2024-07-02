
import os,sys
import numpy as np
import pandas as pd

from Utils import convert_multilabel
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import warnings
warnings.filterwarnings("ignore")

# -- default dim = 2
# -- default method = tsne among 4 type ['pca', 'tsne', 'umap', 'dbscan']
cluster_method = 'tsne' # tsne was used in paper
clustered_x = []
clustered_y = []
clustered_pred_label =[]
clustered_test_label = []
iteration = 0

cancer_types = ['Br', 'Cl', 'Li', 'Lu', 'Pa']

def file_load(file_dir):
	'''
	:param file_dir: prediction& ground truth files directory
	:return: model predictions , ground truth, clustering data save directory
	'''
	# --- load saved files (predictions, ground truth), define save directory
	save_path_predict = file_dir + 'svm' + '_predict/'
	save_path_label = file_dir + 'svm' + '_label/'

	predict_file = save_path_predict + str(iteration).zfill(3) + '_.npy'
	label_file = save_path_label + str(iteration).zfill(3) + '_.npy'

	with open(predict_file, 'rb') as f:
		predict_data = np.load(f)  # ex) probs for 5 cancers[ 0.1, 0.1, 0.7, 0.05, 0.05]

	with open(label_file, 'rb') as f:
		y_test = np.load(f)     #  ex) answer label 2

	save_path_cluster = file_dir + 'svm' + '_{0}_cluster/'.format(cluster_method)
	if not os.path.exists(save_path_cluster):
	    os.makedirs(save_path_cluster)

	return predict_data, y_test, save_path_cluster



def get_clustered_data(pred_norm, y_test, method='tsne' , dim=2):
	'''
	:param pred_norm: prediction values from file_load() function
	:param y_test: ground truths from file_load() function
	:param method: choose clustering method among 4 types ['pca', 'tsne', 'umap', 'dbscan']
	:param dim: define n-dim for dimension reduction
	:return: clustered data
	'''
	# --- clustering
	if method == 'pca':
		model = PCA(dim)
		visual_data = model.fit_transform(pred_norm)
	elif method == 'tsne':
		model = TSNE(n_components=dim, perplexity=7, early_exaggeration=30, init='pca', n_iter=10000,
		            n_iter_without_progress=500, learning_rate=50)
		visual_data = model.fit_transform(pred_norm)
	elif method == 'umap':
		model = umap.UMAP(n_components=dim, min_dist=0.5, n_neighbors=3)
		visual_data = model.fit_transform(pred_norm)
	elif method == 'dbscan': # tsne used for visualization after BDSCAN
		reduction_model = PCA(dim).fit_transform(pred_norm)
		get_cluster = DBSCAN(eps=0.1, min_samples=10).fit_predict(reduction_model)

		# visual_data = pd.DataFrame(reduction_model, columns=['tsne1', 'tsne2'])

		visual_data = pd.DataFrame(reduction_model, columns=['x', 'y'])
		visual_data['dbscan'] = get_cluster
		visual_data['actual'] = y_test


		outliers_idx = np.where(visual_data['dbscan']==-1)
		pred_norm[outliers_idx]


	return visual_data


def visualize_clustered_data(clustered_data, y_pred, y_test, method='tsne', save=True):
	'''
	:param clustered_data: clustered data (x,y)
	:param y_test: ground truth
	:param method: clustering method
	:param save: True if you want to save clustering figure
	'''
	u_labels = np.unique(y_test)  # 0, 1, 2, 3, 4
	img_cluster_log = save_path_cluster + str(iteration).zfill(3) + '_cluster.png'

	if method != 'dbscan':
		for i in u_labels: #  0,1,2,3,4
			plt.scatter(clustered_data[y_test == i, 0], clustered_data[y_test == i, 1], label=convert_multilabel(i))
			clustered_x.extend(clustered_data[y_test == i, 0])
			clustered_y.extend(clustered_data[y_test == i, 1])

			clustered_pred_label.extend(y_pred[np.where(y_pred == i)])
			clustered_test_label.extend(y_test[np.where(y_test == i)])


		cluster_xy_data = pd.DataFrame([clustered_x, clustered_y,clustered_pred_label,clustered_test_label]).transpose()

	else:
		# sns.scatterplot(x = "tsne1", y = "tsne2",hue = "dbscan",data=clustered_data)
		sns.scatterplot(x="x", y="y", hue="dbscan", data=clustered_data)
		cluster_xy_data = pd.concat([clustered_data['x'], clustered_data['y']]
		                            , axis=1, ignore_index=True)

	plt.legend()
	if save:
		plt.savefig(img_cluster_log)
		cluster_xy_data.to_csv(save_path_cluster + "clustered data.csv", header=["x", "y", "pred","test"])

	plt.show()


if __name__=="__main__":
	Cval =50
	Gval =1.2

	saved_file_dir =  'Result/5types_cancer/Alltrain/LR_SVM_C{0}G{1}/'.format(Cval, Gval)
	# -- 0. load prediction results, true label
	predict_data, y_test, save_path_cluster = file_load(saved_file_dir)
	y_pred = np.argmax(predict_data, axis=1)
	# ---1. get & viz clustering results
	clustered_data = get_clustered_data(predict_data,y_test, method =cluster_method, dim=2)

	visualize_clustered_data(clustered_data, y_pred, y_test, method=cluster_method, save=True)




