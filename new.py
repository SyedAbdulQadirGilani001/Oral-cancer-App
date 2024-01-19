# user upload dataset and features asked by user then train model and predict
# the result
import os # for file operation
import sys # for system operation
import time # for time operation
import json # for json operation
import streamlit as st # for web app
import pandas as pd # for dataframe operation
import numpy as np # for array operation
import matplotlib.pyplot as plt # for plot operation
import seaborn as sns # data visualization
import plotly.express as px # interactive data visualization
import plotly.graph_objects as go # interactive data visualization, scatter plot
import plotly.figure_factory as ff # interactive data visualization, histogram
from sklearn.preprocessing import StandardScaler # for data preprocessing
from sklearn.model_selection import train_test_split # for data preprocessing
from sklearn.linear_model import LogisticRegression # for model training
from sklearn.ensemble import RandomForestClassifier # for model training
from sklearn.svm import SVC # for model training
from sklearn.metrics import confusion_matrix # for model evaluation
from sklearn.metrics import accuracy_score # for model evaluation
from sklearn.metrics import classification_report # for model evaluation
from sklearn.metrics import roc_curve # for model evaluation
from sklearn.metrics import roc_auc_score # for model evaluation
from sklearn.metrics import precision_recall_curve # for model evaluation
from sklearn.metrics import auc # for model evaluation
from sklearn.metrics import f1_score # for model evaluation
from sklearn.metrics import average_precision_score # for model evaluation

# set page title
st.set_page_config(page_title='Days to Death Oral Cancer Prediction', layout='wide', initial_sidebar_state='auto', menu_items=None)

# set page title
st.title('Days to Death Oral Cancer Prediction')

# set page subtitle
st.markdown('''
This app predicts the **Alive** or **Death** of Oral Cancer Patients!
* **Python libraries:** pandas, numpy, matplotlib, seaborn, plotly, sklearn
''')

# set sidebar
st.header('User Input Features')
# user upload dataset
uploaded_file = st.file_uploader('Upload your input CSV file', type=['csv'])
# # user select features
# df= pd.read_csv(uploaded_file)
# # head()
# st.write('**Input DataFrame**')
# # dropna
# df = df.dropna(axis=0, how='all')
# df=df.dropna(axis=1,how='all')
# df.fillna(df.mode().iloc[0], inplace=True)
# st.dataframe(df.head())
# # user select features
# selected_features = st.multiselect('Select Features', df.columns)
# # display selected features
# st.write('Selected features', selected_features)
# # display shape of dataset
# st.write('Rows and columns of dataset', df.shape)
# if user upload dataset
if uploaded_file is not None:
    # read dataset
    df = pd.read_csv(uploaded_file)
    # head()
    st.write('**Input DataFrame**')
    # dropna
    df = df.dropna(axis=0, how='all')
    df=df.dropna(axis=1,how='all')
    df.fillna(df.mode().iloc[0], inplace=True)
    st.dataframe(df.head())
    # user select features
    selected_features = st.multiselect('Select Features', df.columns)
    # display selected features
    st.write('Selected features', selected_features)
    # display shape of dataset
    st.write('Rows and columns of dataset', df.shape)
    # display statistics
    st.write('Statistics of dataset', df.describe())
    # hide this and if user select features then display model evaluation
    if selected_features:
        # display model evaluation
        st.subheader('Model Evaluation')
        # algorithm selection
        algorithm = st.selectbox('Select Algorithm', ['Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Naive Bayes', 'Decision Tree'])
        # split dataset into X and y
        X = df[selected_features]
        X['diagnoses/0/ajcc_pathologic_stage'].replace({'Stage I':1,'Stage IA':1,'Stage IB':1,'Stage II':2,'Stage IIA':2,'Stage IIB':2,'Stage III':3,'Stage IIIA':3,'Stage IIIB':3,'Stage IIIC':3,'Stage IV':4,'Stage IVA':4,'Stage IVB':4,'Stage IVC':4,'Not Reported':0},inplace=True)
        y = df['demographic/vital_status']
        # split dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        # data preprocessing
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # model training
        if algorithm == 'Logistic Regression':
            model = LogisticRegression()
        elif algorithm == 'Random Forest':
            model = RandomForestClassifier()
        elif algorithm == 'Support Vector Machine':
            model = SVC()
        model.fit(X_train, y_train)
        # model evaluation
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        # display model evaluation
        # st.write('Confusion Matrix', cm)
        # % accuracy score
        st.write('Accuracy Score %', acc*100)
        # classification report
        # st.write('Classification Report', cr)
        # ROC Curve
        # st.write('ROC Curve')
        # # calculate roc curve
        # ns_fpr, ns_tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        # # plot roc curve
        # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Logistic Regression')
        # # axis labels
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # # show the legend
        # plt.legend()
        # actual and predicted alive or death
        st.write('Actual and Predicted Alive or Death')
        # create dataframe
        df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        # display dataframe
        st.dataframe(df1)
        fig = px.histogram(df, x="demographic/vital_status", color="demographic/vital_status", title="Alive or Death")
    st.plotly_chart(fig)
    # actual and predicted plotly factory bar graph
    fig = go.Figure(data=[
        go.Bar(name='Actual', x=y_test, y=y_test),
        go.Bar(name='Predicted', x=y_test, y=y_pred)
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)
    # seaborn actual and predicted histogram
    fig, ax = plt.subplots()
    sns.histplot(data=y_test, ax=ax, label="Actual", color="green")
    sns.histplot(data=y_pred, ax=ax, label="Predicted", color="red")
    ax.legend()
    st.pyplot(fig)

    death_rate = df[df['demographic/vital_status'] == 'Dead'].shape[0] / df.shape[0]
    survival_rate = df[df['demographic/vital_status'] == 'Alive'].shape[0] / df.shape[0]
    st.write("**Percentage of Dead Patients:**")
    st.write(str(death_rate * 100) + "%")
    # percentage of Not Reported
    st.write("**Percentage of Not Reported Patients:**")
    st.write(str((1 - death_rate - survival_rate) * 100) + "%")
    st.write("**Percentage of Alive Patients:**")
    st.write(str(survival_rate * 100) + "%")
    # sunburst plot of death rate and survival rate
    fig = px.sunburst(df, path=['demographic/vital_status'], title="Alive or Death", width=600, height=600, color_discrete_sequence=['#1E90FF', '#FF4500'])
    st.plotly_chart(fig)
