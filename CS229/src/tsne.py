from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/data/paintersbynumbers/'
feature_dir = 'features/'

X = np.load(feature_dir + 'features_train256_100.npy')
Y = np.load(feature_dir + 'labels_train256_100.npy')

plt.figure()
plt.hold(True)
## plot tsne for painters 0:4
for p in range(5):
	X_p = X[np.where(Y==p)]
	dim1 = X_p.shape[0]
	dim2 = X_p.shape[1]*X_p.shape[2]*X_p.shape[3]
	X_p = X_p.reshape(dim1, dim2)

	print X_p.shape

	#PCA for dimension reduction
	pca = PCA(n_components=50)
	X_pca = pca.fit_transform(X_p)

	#TSNE
	model_tsne = TSNE(n_components=2, random_state=0)
	X_tsne = model_tsne.fit_transform(X_pca)

	plt.scatter(X_tsne[:,0],X_tsne[:,1])
	
plt.show()
