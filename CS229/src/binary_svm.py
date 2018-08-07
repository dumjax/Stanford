import numpy as np
from sklearn import metrics
pair_dir = '/data/paintersbynumbers/'

test_pairs = np.load(pair_dir + 'test_pairs_64.npy')
paintings = np.load(pair_dir + 'features/paintings_test64_finetuned.npy')
predictions = np.load(pair_dir + 'features/svm_predictions.npy')
y_test = np.load(pair_dir + 'y_test_64.npy')
binary_predictions = np.zeros(test_pairs.shape[0])
for idx, pair in enumerate(test_pairs):
	image_name1 = pair[0]
	image_name2 = pair[1]
	idx1 = np.where(paintings == image_name1)
	idx2 = np.where(paintings == image_name2)
	if  predictions[idx1[0][0]]!=predictions[idx2[0][0]]:
		binary_predictions[idx] = 1
	else:
		binary_predictions[idx] = 0

np.save('binary_prediction.npy',binary_predictions)
print y_test[:10], binary_predictions[:10]
precision = metrics.precision_score(y_test, binary_predictions)
accuracy = metrics.accuracy_score(y_test, binary_predictions)
recall = metrics.recall_score(y_test, binary_predictions)
roc_auc = metrics.roc_auc_score(y_test, binary_predictions)
f1_score = metrics.f1_score(y_test, binary_predictions)

print('Accuracy for binary svm: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('roc_auc', roc_auc)
print('fi_score',f1_score)






