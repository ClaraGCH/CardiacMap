# The cardiac epithelium map

### Pipeline:
![fig1 scMorphometrics pipeline](https://github.com/user-attachments/assets/3851c863-c9bf-4e12-899a-5fd1bd18b3ae)
Step 1: Whole mount immunofluorescence of the Dorsal pericardial wall (the cardiac epithelium) with phalloidin to label the cell membranes and any other interesting protein.
Step 2: Segmentation using Tissue Analyser (TA) (Aigouy et al.2016)
Step 3: Quantification of features using TA, Force inference and Dproj
Step 4: Store,clean and merge the data ready to pass Machine learning algorithms.
Step 5: Data analysis

### Goal: 
Understand the patterning of cardiac progenitors.The embryonic heart grows by the progressive addition of progenitor cells located on an epithelial layer known as the Dorsal pericardial wall (DPW).
T-box transcription genes (Tbx1 and Tbx5) play a key role in regulating the addition of the cells to the poles of the heart. However, the mechanisms by which the cells contribute to each pole remain unclear.

### To investigate this, we have:
1. Employed unsupervised clustering algorythms to map the cells according to their morphological features.
2. Applied supervised classification algorythms to predict whether a cell will express Tbx5 gene.

### Final dashboard version: 
https://www.behance.net/gallery/196792557/Cardiac-map-web-dasboard


