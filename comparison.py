# Programmed By Naveen Venkat
# nav.naveenvenkat@gmail.com
# Birla Institute of Technology and Science, Pilani

import numpy as np
import config
from time import time

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# ==========================
# SUBJECT DEPENDENT ANALYSIS
# ==========================

# GET RAW TRAIN AND TEST DATA
def getSubjectDependentRawTrainTest(datasetFile, trainSubjects, testSubjects):
	print
	print "getSubjectDependentRawTrainTest starting ..."

	t = time()
	dataLines = []

	with open(config.datasetFile) as f:
		dataLines = f.readlines()

	train = {}
	test = {}

	for i, a in enumerate(dataLines):

		content = list(int(float(b)) for b in a.split(','))

		subject = content[0]
		video = content[1]

		if subject in trainSubjects:

			if subject not in train:
				train[subject] = []
			else:
				train[subject].append(content)

		elif subject in testSubjects:

			if subject not in test:
				test[subject] = []
			else:
				test[subject].append(content)
	t = time() - t
	print "~ completed in", t, "sec"

	return (train, test)


# SPLIT INTO INPUT AND OUTPUT FOR EACH SUBJECT
def splitSubjectDependentTrainTestIO(train, test, inputColumns, targetColumn):
	
	traininputs = {}
	trainoutputs = {}
	testinputs = {}
	testoutputs = {}

	for subject in train:
		for line in train[subject]:
			
			trainInputData = []
			trainOutputData = []

			for idx in inputColumns:
				trainInputData.append(line[idx-1])

			for idx in targetColumn:
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

			for idx in inputColumns:
				testInputData.append(line[idx-1])

			for idx in targetColumn:
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

	return (traininputs, trainoutputs, testinputs, testoutputs)


# TRAIN EACH CLASSIFIER
def fitSubjectDependentClassifiers(traininputs, trainoutputs, classifiersToTrain=['svm','gnb','ann','knn'], parameters=config.defaultParams):
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

		if 'svm' in classifiersToTrain:
			print "svm ...",
			t = time()
			svm[subject] = SVC(**parameters['svm'])
			svm[subject].fit(X,y)
			t = time() - t
			print "done in", t, "sec"

		if 'gnb' in classifiersToTrain:
			print "gnb ...",
			t = time()
			gnb[subject] = GaussianNB(**parameters['gnb'])
			gnb[subject].fit(X,y)
			t = time() - t
			print "done in", t, "sec"

		if 'ann' in classifiersToTrain:
			print "ann ...",
			t = time()
			ann[subject] = MLPClassifier(**parameters['ann'])
			ann[subject].fit(X,y)
			t = time() - t
			print "done in", t, "sec"

		if 'knn' in classifiersToTrain:
			print "knn ...",
			t = time()
			knn[subject] = KNeighborsClassifier(**parameters['knn'])
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
def predictSubjectDependentClassifiers(svm, gnb, ann, knn, testinputs, testoutputs):
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

		analysis[subject]['svm'] = {'correct':'Not Tested', 'incorrect':'Not Tested', 'accuracy':'Not Tested'}
		analysis[subject]['gnb'] = {'correct':'Not Tested', 'incorrect':'Not Tested', 'accuracy':'Not Tested'}
		analysis[subject]['ann'] = {'correct':'Not Tested', 'incorrect':'Not Tested', 'accuracy':'Not Tested'}
		analysis[subject]['knn'] = {'correct':'Not Tested', 'incorrect':'Not Tested', 'accuracy':'Not Tested'}

		if svm!={}:
			print "svm ...",
			analysis[subject]['svm'] = {'correct':0, 'incorrect':0, 'accuracy':0}
			t = time()
			for i,a in enumerate(testinputs[subject]):
				for trainedSubjects in svm:
					if svm[trainedSubjects].predict([a])[0] == testoutputs[subject][i]:
						analysis[subject]['svm']['correct'] += 1
					else:
						analysis[subject]['svm']['incorrect'] += 1
			t = time() - t
			print "done in", t, "sec"

		if gnb!={}:
			print "gnb ...",
			analysis[subject]['gnb'] = {'correct':0, 'incorrect':0, 'accuracy':0}
			t = time()
			for i,a in enumerate(testinputs[subject]):
				for trainedSubjects in gnb:
					if gnb[trainedSubjects].predict([a])[0] == testoutputs[subject][i]:
						analysis[subject]['gnb']['correct'] += 1
					else:
						analysis[subject]['gnb']['incorrect'] += 1
			t = time() - t
			print "done in", t, "sec"

		if ann!={}:
			print "ann ...",
			analysis[subject]['ann'] = {'correct':0, 'incorrect':0, 'accuracy':0}
			t = time()
			for i,a in enumerate(testinputs[subject]):
				for trainedSubjects in ann:
					if ann[trainedSubjects].predict([a])[0] == testoutputs[subject][i]:
						analysis[subject]['ann']['correct'] += 1
					else:
						analysis[subject]['ann']['incorrect'] += 1
			t = time() - t
			print "done in", t, "sec"

		if knn!={}:
			print "knn ...",
			analysis[subject]['knn'] = {'correct':0, 'incorrect':0, 'accuracy':0}
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
			if analysis[subject][cfier]['accuracy']!='Not Tested':
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
def publishSubjectDependentResults(analysis, trainSubjects, testSubjects, inputColumns, targetColumn, analysisFileExtension=config.analysisFileExtension):
	
	analysisDictFile = 'unnamed_analysisDictFile'
	analysisRawFile = 'unnamed_analysisRawFile'

	if analysisFileExtension is not None:
		analysisDictFile = config.analysisDictFile + '_' + analysisFileExtension
		analysisRawFile = config.analysisRawFile + '_' + analysisFileExtension
	else:
		analysisDictFile = config.analysisDictFile
		analysisRawFile = config.analysisRawFile

	print
	print "writing raw dictionary to", analysisDictFile

	t1 = time()
	
	with open(analysisDictFile, 'a') as f:
		f.write("\n")
		f.write(str(analysis))
	
	t2 = time()
	print "done in", t2-t1, "sec"

	print "writing results to analysis.txt"

	t1 = time()
	with open(analysisRawFile, 'a') as f:
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


# GET CUMULATIVE RESULTS
def getSubjectDependentCumulativeScores(scores, analysis):

	for subject in analysis:
		for cfier in scores:
			if analysis[subject][cfier]['accuracy']!='Not Tested':
				if scores[cfier]=='Not Tested':
					scores[cfier]=0
				scores[cfier] += analysis[subject][cfier]['accuracy']

	return scores


# PUBLISH CUMULATIVE RESULTS
def publishSubjectDependentCumulativeResults(scores, analysisFileExtension=config.analysisFileExtension):

	analysisCumulativeFile = 'unnamed_analysis_file'

	if analysisFileExtension is not None:
		analysisCumulativeFile = config.analysisCumulativeFile + '_' + analysisFileExtension
	else:
		analysisCumulativeFile = config.analysisCumulativeFile

	print "writing results to", analysisCumulativeFile

	t1 = time()
	with open(config.analysisCumulativeFile, 'a') as f:
		for cfier in scores:

			f.write("inputColumns: " + str(inputColumns) + "\n")
			f.write("targetColumn: " + str(targetColumn) + "\n")

			
			f.write(str(cfier) + " : " + str(scores[cfier]) + "\n")
			f.write("-----------------------------------------\n")

	t2 = time()
	print "done in ", t2-t1, "sec"


# RUN SUBJECT DEPENDENT OPERATIONS
def runSubjectDependentOperations(trainSubjects, testSubjects, inputColumns, targetColumn, classifiersToTrain=['svm','gnb','ann','knn'], parameters=config.defaultParams):
	print "starting subject dependent operations"
	startTime = time()

	(train, test) = getSubjectDependentRawTrainTest(config.datasetFile, trainSubjects, testSubjects)
	(traininputs, trainoutputs, testinputs, testoutputs) = splitSubjectDependentTrainTestIO(train, test, inputColumns, targetColumn)
	(svm, gnb, ann, knn) = fitSubjectDependentClassifiers(traininputs, trainoutputs, classifiersToTrain, parameters)
	analysis = predictSubjectDependentClassifiers(svm, gnb, ann, knn, testinputs, testoutputs)
	publishSubjectDependentResults(analysis, trainSubjects, testSubjects, inputColumns, targetColumn)

	endTime = time()
	print
	print "done operations in", endTime - startTime
	return analysis


def exploreKNN():

	pass

# EXECUTION

if __name__ == '__main__':

	scores = {}
	scores['svm'] = 'Not Tested'
	scores['gnb'] = 'Not Tested'
	scores['ann'] = 'Not Tested'
	scores['knn'] = 'Not Tested'

	for i in config.subjects:
		testSubjects = [i]
		trainSubjects = sorted(list(set(config.subjects) - set(testSubjects)))
		print "test", i
		print "train", trainSubjects
		inputColumns = config.inputColumns
		targetColumn = config.targetColumn

		analysis = runSubjectDependentOperations(trainSubjects, testSubjects, inputColumns, targetColumn, config.classifiersToTrain, config.parameters)
		print
		print "GOT ANALYSIS AS", analysis
		print 
		scores = getSubjectDependentCumulativeScores(scores, analysis)
	
	for cfier in scores:
		if scores[cfier]!='Not Tested':
			scores['svm'] /= 9
			scores['gnb'] /= 9
			scores['ann'] /= 9
			scores['knn'] /= 9
	publishSubjectDependentCumulativeResults(scores)


# ===========================================================================================================================
