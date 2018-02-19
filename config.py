# Programmed By Naveen Venkat
# nav.naveenvenkat@gmail.com
# Birla Institute of Technology and Science, Pilani

'''

Column information
------------------

Column 1: Subject ID
Column 2: Video ID
Column 3: Attention (Proprietary measure of mental focus)
Column 4: Mediation (Proprietary measure of calmness)
Column 5: Raw (Raw EEG signal)
Column 6: Delta (1-3 Hz of power spectrum)
Column 7: Theta (4-7 Hz of power spectrum)
Column 8: Alpha 1 (Lower 8-11 Hz of power spectrum)
Column 9: Alpha 2 (Higher 8-11 Hz of power spectrum)
Column 10: Beta 1 (Lower 12-29 Hz of power spectrum)
Column 11: Beta 2 (Higher 12-29 Hz of power spectrum)
Column 12: Gamma 1 (Lower 30-100 Hz of power spectrum)
Column 13: Gamma 2 (Higher 30-100 Hz of power spectrum)
Column 14: predefined label (whether the subject is expected to be confused) - observed by observers
Column 15: user-defined label (whether the subject is actually confused) - confusion reported by the subject

'''

datasetFile = './dataset/EEG data.csv'
analysisDictFile = './analysis/analysis.dict'
analysisRawFile = './analysis/analysis.txt'
analysisCumulativeFile = './analysis/cumulative_analysis.txt'

subjects = [1,2,3,4,5,6,7,8,9]

trainSubjects = [1,2,3,4,5,6,7,8]
testSubjects = [9]

inputColumns = [3,4,5,6,7,8,9,10,11,12,13,14]
targetColumn = [15]
