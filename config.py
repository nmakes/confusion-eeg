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
Column 14: predefined label (whether the subject is expected to be confused)
"Additionally,  there  were  three  student  observers 
watching the body-language of the student. Each observer rated the confusion level of 
the student in each session on a scale of 1-7. The conventional scale of 1-7 was used.
Four observers were asked  to  observe  1-8  students each,  so  that  there  was  not  an  ef-
fect of observers just studying one student."
Column 15: user-defined label (whether the subject is actually confused)

'''

# Select columns for input and output

inputColumns = [3,4,6,7,8,9,10,11,12,13]
targetColumn = 15