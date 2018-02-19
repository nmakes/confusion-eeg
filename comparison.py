import numpy as np
import config
from time import time

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
		
		trainInputData = []
		trainOutputData = []

		for idx in config.inputColumns:
			trainInputData.append(line[idx-1])

		for idx in config.targetColumn:
			trainOutputData.append(line[idx-1])
		
		if subject in traininputs:
			traininputs[subject].append(trainInputData)
		else:
			traininputs[subject] = []
			traininputs[subject].append(trainInputData)
		
		if subject in trainoutputs:
			trainoutputs[subject].append(trainOutputData)
		else:
			trainoutputs[subject] = []
			trainoutputs[subject].append(trainOutputData)


for subject in test:
	for line in test[subject]:
		
		testInputData = []
		testOutputData = []

		for idx in config.inputColumns:
			testInputData.append(line[idx-1])

		for idx in config.targetColumn:
			testOutputData.append(line[idx-1])
		
		if subject in testinputs:
			testinputs[subject].append(testInputData)
		else:
			testinputs[subject] = []
			testinputs[subject].append(testInputData)
		
		if subject in testoutputs:
			testoutputs[subject].append(testOutputData)
		else:
			testoutputs[subject] = []
			testoutputs[subject].append(testOutputData)


# TRAIN EACH CLASSIFIER

def fitClassifiers(traininputs, trainoutputs):
	print
	print "fitClassifiers starting ..."
	print "number of train subjects =", len(traininputs.keys())

	runningTime = time()

	svm = {}
	gnb = {}
	ann = {}
	knn = {}

	for subject in traininputs:

		print
		print "Training for subject", subject

		T = time()

		X = np.array(traininputs[subject])
		y = np.array(trainoutputs[subject])
		l = len(y)
		y = y.reshape(l,)

		print "svm ...",
		t = time()
		svm[subject] = SVC()
		svm[subject].fit(X,y)
		t = time() - t
		print "done in", t, "sec"

		print "gnb ...",
		t = time()
		gnb[subject] = GaussianNB()		
		gnb[subject].fit(X,y)
		t = time() - t
		print "done in", t, "sec"

		print "ann ...",
		t = time()
		ann[subject] = MLPClassifier(learning_rate = 'adaptive')
		ann[subject].fit(X,y)
		t = time() - t
		print "done in", t, "sec"

		print "knn ...",
		t = time()
		knn[subject] = KNeighborsClassifier()
		knn[subject].fit(X,y)
		t = time() - t
		print "done", t, "sec"

		T = time() - T
		print "~ completed subject", subject, "in", T, "sec"

	runningTime = time() - runningTime

	print
	print "done fitting classifiers in", runningTime
	print
	print "----------------------------------"
	return (svm, gnb, ann, knn)


# PREDICTION BY EACH CLASSIFIER

def predictClassifiers(svm, gnb, ann, knn, testinputs, testoutputs):
	print
	print "predictClassifiers starting ..."
	print "number of test subjects =", len(testinputs.keys())

	runningTime = time()

	analysis = {}

	for subject in testinputs:

		print
		print "Testing for subject", subject
		T = time()

		analysis[subject] = {}

		analysis[subject]['svm'] = {'correct':0, 'incorrect':0, 'accuracy':0}
		analysis[subject]['gnb'] = {'correct':0, 'incorrect':0, 'accuracy':0}
		analysis[subject]['ann'] = {'correct':0, 'incorrect':0, 'accuracy':0}
		analysis[subject]['knn'] = {'correct':0, 'incorrect':0, 'accuracy':0}

		print "svm ...",
		t = time()
		for i,a in enumerate(testinputs[subject]):
			for trainedSubjects in svm:
				if svm[trainedSubjects].predict([a])[0] == testoutputs[subject][i]:
					analysis[subject]['svm']['correct'] += 1
				else:
					analysis[subject]['svm']['incorrect'] += 1
		t = time() - t
		print "done in", t, "sec"

		print "gnb ...",
		t = time()
		for i,a in enumerate(testinputs[subject]):
			for trainedSubjects in gnb:
				if gnb[trainedSubjects].predict([a])[0] == testoutputs[subject][i]:
					analysis[subject]['gnb']['correct'] += 1
				else:
					analysis[subject]['gnb']['incorrect'] += 1
		t = time() - t
		print "done in", t, "sec"

		print "ann ...",
		t = time()
		for i,a in enumerate(testinputs[subject]):
			for trainedSubjects in ann:
				if ann[trainedSubjects].predict([a])[0] == testoutputs[subject][i]:
					analysis[subject]['ann']['correct'] += 1
				else:
					analysis[subject]['ann']['incorrect'] += 1
		t = time() - t
		print "done in", t, "sec"

		print "knn ...",
		t = time()
		for i,a in enumerate(testinputs[subject]):
			for trainedSubjects in knn:
				if knn[trainedSubjects].predict([a])[0] == testoutputs[subject][i]:
					analysis[subject]['knn']['correct'] += 1
				else:
					analysis[subject]['knn']['incorrect'] += 1
		t = time() - t
		print "done in", t, "sec"

		for cfier in analysis[subject]:
			analysis[subject][cfier]['accuracy'] = float(analysis[subject][cfier]['correct']) / float(analysis[subject][cfier]['correct'] + analysis[subject][cfier]['incorrect']) * 100

		T = time() - T
		print "~ completed subject", subject, "in", T, "sec"

	runningTime = time() - runningTime

	print
	print "done predicting classifiers in", runningTime
	print
	print "----------------------------------"

	return analysis


# PUBLISH RESULTS

def publishResults(analysis, trainSubjects, testSubjects, inputColumns, targetColumn):
	print
	print "writing raw dictionary to analysis.dict"

	t1 = time()
	
	with open('analysis.dict', 'w+') as f:
		f.write(str(analysis))
	
	t2 = time()
	print "done in", t2-t1, "sec"

	print "writing results to analysis.txt"

	t1 = time()
	with open('analysis.txt', 'w+') as f:
		for testSubject in analysis:

			f.write("\n")
			f.write("-----------------------------------------\n")
			f.write("trainSubjects: " + str(trainSubjects) + "\n")
			f.write("testSubjects: " + str(testSubjects) + "\n")
			f.write("inputColumns: " + str(inputColumns) + "\n")
			f.write("targetColumn: " + str(targetColumn) + "\n")

			for cfier in analysis[testSubject]:
				f.write(str(cfier) + " : " + str(analysis[testSubject][cfier]['accuracy']) + "\n")
			f.write("-----------------------------------------\n")
	t2 = time()
	print "done in ", t2-t1, "sec"


print "starting operations"
startTime = time()

(svm, gnb, ann, knn) = fitClassifiers(traininputs, trainoutputs)
analysis = predictClassifiers(svm, gnb, ann, knn, testinputs, testoutputs)
publishResults(analysis, config.trainSubjects, config.testSubjects, config.inputColumns, config.targetColumn)

endTime = time()
print
print "done operations in", endTime - startTime