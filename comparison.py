import numpy as np
import config

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

x = []
file = 'EEG data.csv'
with open(file) as f:
	x = f.readlines()

train = []
test = []
traininput = []
trainoutput = []
testinput = []
testoutput = []

columnToLearn = 15
labelToLearn = columnToLearn - 1

trainTestSplit = 0.99
trainSize = int(trainTestSplit * len(x))

print
print "Data Size =", len(x)
print "Train =", trainSize
print "Test =", len(x) - trainSize
print

for i, a in enumerate(x):
	if i < trainSize:
		train.append(list(int(float(b)) for b in a.split(',')))
	else:
		test.append(list(int(float(b)) for b in a.split(',')))


for i, a in enumerate(train):
	trainoutput.append(a[labelToLearn])
	traininput.append(a[2:13])


for i, a in enumerate(test):
	testoutput.append(a[labelToLearn])
	testinput.append(a[2:13])


X = np.array(traininput)
y = np.array(trainoutput)


#SVM Classification Training

print "SVM running ...",

svm = SVC()
svm.fit(X, y)

print "done"

#Gaussian Naive Bayes

print "GNB running ...",

gnb = GaussianNB()
gnb.fit(X, y)

print "done"

#Artificial Neural Network

print "ANN running ...",

ann = MLPClassifier(learning_rate = 'adaptive')
ann.fit(X, y)

print "done"

#K-Nearest Neighbor Classifier

print "KNN running ...",

knn = KNeighborsClassifier()
knn.fit(X, y)

print "done"

def analyze(svm, gnb, ann, knn, testinput, testoutput):
	
	analysis = {}

	correct = [0, 0, 0, 0]
	incorrect = [0, 0, 0, 0]

	analysis['svm'] = {'correct':0, 'incorrect':0, 'accuracy':0}
	analysis['gnb'] = {'correct':0, 'incorrect':0, 'accuracy':0}
	analysis['ann'] = {'correct':0, 'incorrect':0, 'accuracy':0}
	analysis['knn'] = {'correct':0, 'incorrect':0, 'accuracy':0}

	for i, a in enumerate(testinput):
		
		if svm.predict([a])[0] == testoutput[i]:
			analysis['svm']['correct'] += 1
		else:
			analysis['svm']['incorrect'] += 1
		
		if gnb.predict([a])[0] == testoutput[i]:
			analysis['gnb']['correct'] += 1
		else:
			analysis['gnb']['incorrect'] += 1
		
		if ann.predict([a])[0] == testoutput[i]:
			analysis['ann']['correct'] += 1
		else:
			analysis['ann']['incorrect'] += 1
		
		if knn.predict([a])[0] == testoutput[i]:
			analysis['knn']['correct'] += 1
		else:
			analysis['knn']['incorrect'] += 1

	for key in analysis:
		analysis[key]['accuracy'] = float(analysis[key]['correct']) / float(analysis[key]['correct'] + analysis[key]['incorrect']) * 100

	return analysis


print "ANALYSIS running ...",
analysis = analyze(svm, gnb, ann, knn, testinput, testoutput)
print "done"
print
# print "SVM =", float(correct[0])/float(incorrect[0]+correct[0]) * 100
# print "GNB =", float(correct[1])/float(incorrect[1]+correct[1]) * 100
# print "ANN =", float(correct[2])/float(incorrect[2]+correct[2]) * 100
# print "KNN =", float(correct[3])/float(incorrect[3]+correct[3]) * 100

for key in analysis:
	print key, ":"
	for k2 in analysis[key]:
		print k2, "=", analysis[key][k2]
	print