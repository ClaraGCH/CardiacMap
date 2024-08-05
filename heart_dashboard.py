import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import math
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
import xgboost 
from sklearn.metrics import accuracy_score, f1_score, RocCurveDisplay, ConfusionMatrixDisplay, make_scorer
from sklearn.ensemble import (RandomForestClassifier,
                              BaggingClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier,
                              VotingClassifier,
                              StackingClassifier)
## CONFIG
st.set_page_config(
    page_title="Heart cells",
    page_icon="❤️🦠",
    layout="wide",
    initial_sidebar_state="expanded")

### SIDE BAR
st.sidebar.header("Sections")
st.sidebar.markdown("""
    * [Welcome to the cells that will give rise to your heart!](#welcome-to-the-cells-that-will-give-rise-to-your-heart)
    * [Presentation of the data](#presentation-of-the-data)
    * [Data preprocessing](#data-preprocessing)
    * [Unsupervised learning : Cell clustering](#unsupervised-learning-cell-clustering)
    * [Supervised learning : Cell classification](#supervised-learning-cell-classification)
    * [Tbx5 prediction App](#tbx5-prediction-App)               
""")

### TITLE AND TEXT
st.markdown("# Heart cells patterning")
st.subheader("Welcome to the cells that will give rise to your heart!")
st.subheader("You can have insights on how these cells look like below👇")

##### PUT VIDEO
st.image("video_gif.gif", caption="Landing on the Dorsal paricardial wall cells 🦠🦠 ", use_column_width=True)

### LOAD DATA
dfE9 = pd.read_csv('E9WT_forclassification.csv')
dfE9 = dfE9.dropna()
dfE9 = dfE9.rename (columns={
    'theta1N': 'stress direction',
    'orientation_cells': 'orientation',
    'area_cells': 'area',
    'perimeter_length': 'perimeter',
    'pressures': 'pressure',
    'eccentricity': 'eccentricity',
    'ratio_lambda': 'stress anisotropy',
    'n_neighbors': 'neighbors',
    'ATAN2': 'golgi polarity'})
dfE9['Tbx5'] = pd.to_numeric(dfE9['Tbx5'], errors='coerce')
dfE9['stress direction'] = dfE9['stress direction'].apply(lambda x: x * (180 / math.pi))
dfE9['orientation'] = dfE9['orientation'].apply(lambda x: x * (180 / math.pi))
dfE9['area'] = dfE9['area']*0.32
dfE9['perimeter'] = dfE9['perimeter']*0.16
dfE9['golgi polarity'] = np.degrees(dfE9['golgi polarity'])

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(dfE9) 

#Data for the PCA dimentionality reduction
dfE9_1 = dfE9.drop(columns=['local_id_cells','Nb_embryo','Tbx5','type']) #'center_x_cells', 'center_y_cells',
st.subheader("Presentation of the data")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Distribution of features in the tissue space")
    embryo = st.selectbox("Select an embryo ", dfE9["Nb_embryo"].sort_values().unique())
    embryones = dfE9[dfE9["Nb_embryo"] == embryo]
    # Plot the selected embryo
    fig = px.scatter(embryones, x="center_x_cells", y="center_y_cells", 
                     title=f"Scatter plot of cells in the real space for Embryo {embryo}")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("Distribution of features in the PCA space")
    X_normalized = StandardScaler().fit_transform(dfE9_1)
    pca = PCA(n_components=2)  # Keep only 2 principal components
    pca_array = pca.fit_transform(X_normalized)
    pca_df = pd.DataFrame(pca_array, columns=["PCA1", "PCA2"])
    pca_df["Tbx5"] = dfE9["Tbx5"].values# Add 'Tbx5' column to PCA DataFrame
    
    fig = px.scatter(pca_df, x="PCA1", y="PCA2",
                     title="Scatter plot of cells in PCA space",
                     width=500, height=500,
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Data preprocessing")
col1, col2, col3 = st.columns(3)
with col1:
    df_1_normalized = StandardScaler().fit_transform(dfE9_1)
    pca = PCA()
    pca.fit(df_1_normalized)
    explained_var = pca.explained_variance_ratio_ * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"],
        y=explained_var,
        text=np.round(explained_var, 2),
        textposition='auto',
        marker=dict(color='blue'),
        name='Explained Variance (%)'))
    
    fig.update_layout(
        title='PCA Explained Variance',
        xaxis_title='Principal Components',
        yaxis_title='Explained Variance (%)',
        width=400,
        height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    pca_array = pca.fit_transform(df_1_normalized)
    abs_pca_components = np.abs(pca.components_)

    fig = go.Figure(data=go.Heatmap(
        z=abs_pca_components,
        x=dfE9_1.columns,
        y=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"],
        colorscale='Viridis'))

    fig.update_layout(
        title='Heatmap of PCA Components',
        xaxis_title='Features',
        yaxis_title='Principal Components',
        xaxis=dict(tickangle=-60),
        width=400,
        height=400)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    corr_matrix = dfE9_1.corr()
    fig = px.imshow(corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='Viridis')
    fig.update_layout(title="Correlation Matrix") #, width=400, height=400
    st.plotly_chart(fig, use_container_width=True)

col1, col2= st.columns(2)
with col1:
    dfE9_2 = dfE9.drop(columns=['center_x_cells', 'center_y_cells', 'local_id_cells','Nb_embryo','Tbx5','type','area'])
    # Perform Standard Scaling
    X_normalized = StandardScaler().fit_transform(dfE9_2)
    df_2_normalized = StandardScaler().fit_transform(dfE9_2)
    pca = PCA()
    pca.fit(df_2_normalized)
    explained_var = pca.explained_variance_ratio_ * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"],
        y=explained_var,
        text=np.round(explained_var, 2),
        textposition='auto',
        marker=dict(color='blue'),
        name='Explained Variance (%)'))
    fig.update_layout(
        title='PCA Explained Variance',
        xaxis_title='Principal Components',
        yaxis_title='Explained Variance (%)',
        width=400,
        height=400)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    pca_array = pca.fit_transform(df_2_normalized)
    abs_pca_components = np.abs(pca.components_)

    fig = go.Figure(data=go.Heatmap(
        z=abs_pca_components,
        x=dfE9_2.columns,
        y=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"],
        colorscale='Viridis'))

    fig.update_layout(
        title='Heatmap of PCA Components',
        xaxis_title='Features',
        yaxis_title='Principal Components',
        xaxis=dict(tickangle=-60),
        width=400,
        height=400)
    st.plotly_chart(fig, use_container_width=True)

######################################################################################################
st.subheader("Unsupervised learning : Cell clustering")
st.markdown("### Do the cells with similar morphological and mechanical features attribute in the space?")

# Perform clustering 
dfE9b = dfE9[['stress direction','orientation']]
pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3))
labels = pipeline.fit_predict(dfE9b)
dfE9['k'] = labels

col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("Distribution of features in the tissue space")
    embryo = st.selectbox("Select an embryo ", dfE9["Nb_embryo"].sort_values().unique(), key="embryo_selectbox_col1")
    embryones = dfE9[dfE9["Nb_embryo"] == embryo]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=embryones, x="center_x_cells", y="center_y_cells", hue='k', palette='Set1')
    plt.title(f"Scatter plot of cells in the real space for Embryo {embryo}")
    plt.gca().invert_yaxis()  # Reverse the y-axis
    plt.xlabel("center_x_cells")
    plt.ylabel("center_y_cells")
    plt.legend(title='k', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    st.pyplot(plt)
    
    
st.markdown("Distribution of the clusters in the tissue space")
embryo = st.selectbox("Select an embryo", dfE9["Nb_embryo"].sort_values().unique(), key="embryo_selectbox")

col1, col2, col3 = st.columns(3)
for index, col in enumerate([col1, col2, col3]):
    with col:
        st.markdown(f"Scatter plot of cells in the real space for Embryo {embryo}")
        embryones = dfE9[dfE9["Nb_embryo"] == embryo]
        attribute_label = index  # attribute labels are 0, 1, and 2
        attribute_data = embryones[embryones['k'] == attribute_label]
        plt.figure(figsize=(10, 6))
        cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}
        plt.scatter(attribute_data["center_x_cells"], attribute_data["center_y_cells"], c=attribute_data["k"].map(cluster_colors))
        plt.title(f"Cluster {attribute_label}")
        plt.gca().invert_yaxis()  # Reverse the y-axis
        plt.xlabel("center_x_cells")
        plt.ylabel("center_y_cells")
        plt.tight_layout()
        st.pyplot(plt)

st.markdown("Distribution of the features for each cluster")
attributes = ['orientation', 'stress direction', 'perimeter', 'area', 'pressure',
              'stress anisotropy', 'eccentricity', 'neighbors', 'golgi polarity']

attribute = st.selectbox("Select a feature", attributes, key="feature_selectbox")

# Check if the selected attribute exists in the dataframe
if attribute in dfE9.columns:
    try:
        fig = px.histogram(dfE9, x=attribute, color="k", 
                           title=f"{attribute} by cluster",
                           color_discrete_map=cluster_colors)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating histogram: {e}")
else:
    st.error(f"The attribute {attribute} does not exist in the dataframe")
st.subheader("Supervised learning : Cell classification")
st.markdown("### Can we predict whether a cell will express Tbx5 from their morphology")
st.markdown("Tbx5+ = 1 / Tbx5- = 0")

col1, col2 = st.columns(2)
with col1:
    st.markdown("Distribution of the Tbx5+ cells in the tissue space")
    embryo = st.selectbox("Select an embryo ", dfE9["Nb_embryo"].sort_values().unique(), key="embryo_selectbox2")
    embryones = dfE9[dfE9["Nb_embryo"] == embryo]
    fig = px.scatter(embryones, x="center_x_cells", y="center_y_cells", color = 'Tbx5',
                     title=f"Scatter plot of cells in the real space for Embryo {embryo}",
                     color_discrete_map=cluster_colors)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("Distribution of the Tbx5+ cells  in the PCA space")
    fig = px.scatter(pca_df, x="PCA1", y="PCA2",
                     title="Scatter plot of cells in PCA space", color = 'Tbx5',
                     color_discrete_map=cluster_colors,
                     width=500, height=500,
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("Distribution of the features in the Tbx5+ or Tbx5- cells")
attribute = st.selectbox("Select a feature", attributes, key="feature_selectbox_for classif")
cluster_colors = {0: 'blue', 1: 'white'}
fig = px.histogram(dfE9, x=attribute, color="Tbx5", 
                           title=f"{attribute} by cell type",
                           color_discrete_map=cluster_colors)
                           #color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig, use_container_width=True)

col1, col2= st.columns(2)
data = {
    'Model': ['Logistic Regression', 'Decission Tree', 'Random Forest', 'Random Forest (FS)', 
        'Non-linear SVM with radial basis function', 'Non-linear SVM with radial basis function (FS)', 
        'Bagging with decision tree as base estimator (FS)', 'Adaboost with decision tree as base estimator', 
        'Boosting with decision tree as base estimator', 'XGBoost'],
    'Accuracy Train':[ 0.793, 0.793, 0.959, 0.956, 0.970, 0.999, 0.986, 0.978, 0.941, 0.993],
    'Accuracy Test': [ 0.781, 0.945, 0.931, 0.925, 0.923, 0.952, 0.953, 0.960, 0.938, 0.974],
    'F1_Score_Train':[0.538, 0.538, 0.869, 0.862, 0.903, 0.996, 0.950, 0.927, 0.763, 0.974],
    'F1_Score_Test': [0.520, 0.805, 0.781, 0.767, 0.750, 0.825, 0.841, 0.866, 0.752, 0.907]}
with col1:
    df_classif_recapitulatif = pd.DataFrame(data)
    st.write(df_classif_recapitulatif)
with col2:
    st.markdown("Feature selection")
    image1 = Image.open ("RF_FeatureSelection.png")
    st.image(image1, use_column_width=True)


col1, col2= st.columns(2)
with col1:
    st.markdown("Confusion matrix on the train set")
    image2 = Image.open ("Matriz_Confusion_Train_XGBoost.png")
    st.image(image2, use_column_width=True)

with col2:
    st.markdown("Confusion matrix on the test set")
    image3 = Image.open ("Matriz_Confusion_Test_XGBoost.png")
    st.image(image3, use_column_width=True)

st.markdown("<div id='tbx5-prediction-App'></div>", unsafe_allow_html=True)
st.subheader("""Tbx5 prediction App""")
st.write("#### This app predicts if a cell is going to express **Tbx5 gene**")
st.sidebar.header ('User Input parameters for Tbx5 cell classification')
def user_input_features():
    center_x_cells = st.sidebar.slider('center_x_cells',0,1000,522)	
    center_y_cells = st.sidebar.slider('center_y_cells',0,1000,745)
    orientation = st.sidebar.slider('orientation',0,180,87)
    perimeter = st.sidebar.slider('perimeter',0,40,25)
    pressure = st.sidebar.slider('pressure',-0.1,0.3,0.001)
    stress_direction  = st.sidebar.slider('stress direction',-90,90,0)
    Nb_embryo = st.sidebar.slider('Nb_embryo',1,6,1)
    data = {'stress direction' : stress_direction,
            'center_y_cells' : center_y_cells,
            'center_x_cells' : center_x_cells,
            'perimeter' : perimeter,
            'orientation' : orientation,
            'pressure' : pressure,
            'Nb_embryo': Nb_embryo}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User input parameters')
st.write(df)

preprocessor = load('preprocessor.pkl')
df_prep = preprocessor.transform(df)
modelXGboost = load('xgboost.joblib')
prediction = modelXGboost.predict(df_prep)
prediction_proba = modelXGboost.predict_proba(df_prep)
st.subheader('Prediction')
st.write(prediction)
st.subheader('Prediction probability')
st.write(prediction_proba)

# st.subheader("Supervised learning : Image classification")
# st.markdown("### Can we predict whether an embryo will be healthy or mutant from an image?")
# col1, col2 = st.columns(2)
# with col1:
#     image1 = Image.open ("CONTROL_E9WT.png")
#     st.image(image1, caption="Control",use_column_width=True)

# with col2:
#     image2 = Image.open ("MUTANT_Tbx5.png")
#     st.image(image2, caption = "Tbx5 mutant")

#### PUT FINAL IMAGE
image = Image.open ("imageDPW.tif")
desired_width = 100  # Adjust this value according to your preference
aspect_ratio = image.width / image.height
desired_height = int(desired_width / aspect_ratio)
image_resized = image.resize((desired_width, desired_height))
st.image(image, caption="Landed on the Dorsal pericardial wall", use_column_width=True)
