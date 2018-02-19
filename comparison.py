import numpy as np
import config

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

x = []
file = 'EEG data.csv'


# Read Contents of the dataset
with open(file) as f:
	x = f.readlines()


# GET RAW TRAIN AND TEST DATA

train = {}
test = {}

for i, a in enumerate(x):

	content = list(int(float(b)) for b in a.split(','))

	subject = content[0]
	video = content[1]

	if subject in config.trainSubjects:

		if subject not in train:
			train[subject] = []
		else:
			train[subject].append(content)

	elif subject in config.testSubjects:

		if subject not in test:
			test[subject] = []
		else:
			test[subject].append(content)


# SPLIT INTO INPUT AND OUTPUT FOR EACH SUBJECT

traininputs = {}
trainoutputs = {}
testinputs = {}
testoutputs = {}

for subject in train:
	for line in train[subject]:
		
		trainData = []

		for idx in config.inputColumns:
			trainData.append(line[idx])
		
		if subject in traininputs:
			traininputs[subject].append(trainData)
		else:
			traininputs[subject] = []
			traininputs[subject].append(trainData)
		
		if subject in trainoutputs:
			trainoutputs[subject].append(trainData)
		else:
			trainoutputs[subject] = []
			trainoutputs[subject].append(trainData)

		trainoutputs.append(line[labelToLearn])


for subject in test:
	for line in test[subject]:
		
		testData = []

		for idx in config.inputColumns:
			testData.append(line[idx])
		
		if subject in testinputs:
			testinputs[subject].append(testData)
		else:
			testinputs[subject] = []
			testinputs[subject].append(testData)
		
		if subject in testoutputs:
			testoutputs[subject].append(testData)
		else:
			testoutputs[subject] = []
			testoutputs[subject].append(testData)

		testoutputs.append(line[labelToLearn])


def fitClassifiers(svm, gnb, ann, knn, traininputs, trainoutputs):

	for subject in traininputs:

		X = np.array(traininputs[subject])
		y = np.array(trainoutputs[subject])

		svm = SVC()
		svm[subject].fit(X,y)

		gnb = GaussianNB()		
		gnb[subject].fit(X,y)

		ann = MLPClassifier(learning_rate = 'adaptive')
		ann[subject].fit(X,y)

		knn = KNeighborsClassifier()
		knn[subject].fit(X,y)

	return svm, gnb, ann, knn


''' UNDER DEVELOPMENT 

def predictClassifiers(svm, gnb, ann, knn, testinputs, testoutputs):
	
	for subject in testinputs:

		analysis = {}

		correct = [0, 0, 0, 0]
		incorrect = [0, 0, 0, 0]

		analysis['svm'] = {'correct':0, 'incorrect':0, 'accuracy':0}
		analysis['gnb'] = {'correct':0, 'incorrect':0, 'accuracy':0}
		analysis['ann'] = {'correct':0, 'incorrect':0, 'accuracy':0}
		analysis['knn'] = {'correct':0, 'incorrect':0, 'accuracy':0}

		for i, a in enumerate(testinputs):
			
			if svm.predict([a])[0] == testoutputs[i]:
				analysis['svm']['correct'] += 1
			else:
				analysis['svm']['incorrect'] += 1
			
			if gnb.predict([a])[0] == testoutputs[i]:
				analysis['gnb']['correct'] += 1
			else:
				analysis['gnb']['incorrect'] += 1
			
			if ann.predict([a])[0] == testoutputs[i]:
				analysis['ann']['correct'] += 1
			else:
				analysis['ann']['incorrect'] += 1
			
			if knn.predict([a])[0] == testoutputs[i]:
				analysis['knn']['correct'] += 1
			else:
				analysis['knn']['incorrect'] += 1

		for key in analysis:
			analysis[key]['accuracy'] = float(analysis[key]['correct']) / float(analysis[key]['correct'] + analysis[key]['incorrect']) * 100

		return analysis


print "ANALYSIS running ...",
analysis = analyze(svm, gnb, ann, knn, testinputs, testoutputs)
print "done"
print

for key in analysis:
	print key, ":"
	for k2 in analysis[key]:
		print k2, "=", analysis[key][k2]
	print

'''