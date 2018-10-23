import sys
import os
import pickle
import numpy as np
from sklearn import svm

def train_data(src, db, exp):
	with open(db,'r') as f:
		pathImages = f.read().strip('\n').split('\n')

	dataX = []; trainY = []

	for path in pathImages:
		pos = path.rfind('\\')
		tag = path[path[:pos].rfind('\\')+1:pos]
		trainY.append(tag)

	if not os.path.exists(exp+"/model.dat"):
		for path in pathImages:
			pos = path.rfind('\\')
			pathFeature = src + '/features/vgg16_fc2/{}.npy'.format(path[path[0:pos].rfind('\\')+1:path.rfind('.')])
			dataX.append(np.load(pathFeature)[0])
		print('Prepare Data: Done!')

		clf = svm.SVC(kernel='linear', C = 1.0)

		print("Training a Linear SVM Classifier")
		clf.fit(dataX, trainY)

		os.makedirs(exp)
		os.chdir(exp)
		pickle.dump(clf, open("model.dat", 'wb'))

		print("Training Successful!")
	print("Finish!")

def main():
	src = sys.argv[1] #E:/MayHocTrongThiGiacMayTinh/deadline/baitap2
	db = sys.argv[2]  #E:/MayHocTrongThiGiacMayTinh/deadline/baitap2/db/db1/train.txt
	exp = sys.argv[3] #E:/MayHocTrongThiGiacMayTinh/deadline/baitap2/exp/svm_linear
	train_data(src, db, exp)

if __name__=='__main__':
	main()