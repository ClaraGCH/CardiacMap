# The cardiac epithelium map

### Pipeline:
![fig1 scMorphometrics Pipeline (2)](https://github.com/user-attachments/assets/6951fcb8-b677-4888-80c1-f5e33ed7e2b1)

- Step 1: Perform Whole mount immunofluorescence of the Dorsal pericardial wall (the cardiac epithelium) with phalloidin to label the cell membranes and any other interesting protein.
- Step 2: Segmentat the images using Tissue Analyser (TA) as described by Aigouy et al.2016.
- Step 3: Quantification of features using TA, Force inference (Kong et al. 2019) and Dproj (Herbert et al. 2021).
- Step 4: Store,clean and merge the data to prepare it for machine learning algorithms.
- Step 5: Analyze the data.

### Goal: 
Understand the patterning of cardiac progenitors.The embryonic heart grows by the progressive addition of progenitor cells located on an epithelial layer known as the Dorsal pericardial wall (DPW).
T-box transcription genes (Tbx1 and Tbx5) play a key role in regulating the addition of the cells to the poles of the heart. However, the mechanisms by which the cells contribute to each pole remain unclear.

### To investigate this, we have:
1. Employed unsupervised clustering algorythms to map the cells according to their morphological features.
2. Applied supervised classification algorythms to predict whether a cell will express Tbx5 gene.

### Final dashboard version: 
https://www.behance.net/gallery/196792557/Cardiac-map-web-dasboard


