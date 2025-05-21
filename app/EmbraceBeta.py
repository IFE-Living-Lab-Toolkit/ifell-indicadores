import pandas as pd
import numpy as np

import seaborn as sns
import datetime
import matplotlib.dates as mdates

import shutil

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import joblib

import streamlit as st

from io import StringIO

st.title("Mi primera app con Streamlit")


def load_dataset(data):
    #dataset = pd.read_csv("C:/Users/ifeex/OneDrive/Documentos/Modelos Predictivos/Modelos Predictivos/Predicción/[FINAL]EEG_Dataset.csv", header=0)
    dataset = pd.read_csv(data)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna()

    dataset['TimeStamp'] = pd.to_datetime(dataset['TimeStamp'])

    return dataset



def load_model(Indicator):
        if (Indicator == "Alertness"):
                rf = joblib.load('C:/Users/ifeex/Downloads/Modelos Predictivos 04252025 (1)/Modelos Predictivos/Modulos/' + Indicator + '/RF94-EP-' + Indicator + '.joblib')
                return rf
        elif (Indicator == "Concentration"):
                rf = joblib.load('C:/Users/ifeex/Downloads/Modelos Predictivos 04252025 (1)/Modelos Predictivos/Modulos/' + Indicator + '/RF97-EP-' + Indicator + '.joblib')
                return rf
        elif (Indicator == "Fatigue"):
               rf = joblib.load('C:/Users/ifeex/Downloads/Modelos Predictivos 04252025 (1)/Modelos Predictivos/Modulos/' + Indicator + '/RF97-EP-' + Indicator + '.joblib')
               return rf
        elif (Indicator == "Motivation"):
               rf = joblib.load('C:/Users/ifeex/Downloads/Modelos Predictivos 04252025 (1)/Modelos Predictivos/Modulos/' + Indicator + '/RF99-EP-' + Indicator + '.joblib')
               return rf
        elif (Indicator == "Stress"):
               rf = joblib.load('C:/Users/ifeex/Downloads/Modelos Predictivos 04252025 (1)/Modelos Predictivos/Modulos/' + Indicator + '/RF99-EP-' + Indicator + '.joblib')
               return rf
        else : 
               print("Hola")


def prediction(rf, dataset, Indicator):

    dataset.columns = dataset.columns.str.lower()
    new_columns = ['user', 'timestamp', 'time', 'section', 'bvp', 'eda', 'temperature']
    dataset = dataset[new_columns]
    dataset = dataset.rename(columns = {'timestamp':'TimeStamp'})
    dataset = dataset.rename(columns = {'section':'Section'})
    if Indicator == 'Alertness':  
        dataset = dataset.rename(columns = {'user':'User'})
        dataset = dataset.rename(columns = {'time':'Time'})
        dataset = dataset.rename(columns = {'bvp':'BVP'})
        dataset = dataset.rename(columns = {'eda':'EDA'})
        dataset = dataset.rename(columns = {'temperature':'Temperature'})
        X = dataset[['BVP', 'EDA', 'Temperature']]
    else:
        X = dataset[['bvp', 'eda', 'temperature']]
    
    y_pred = rf.predict(X)
    dataset[Indicator] = y_pred


    Gamificacion = dataset.loc[dataset['Section']== 1.0]
    TemaConceptual = dataset.loc[dataset['Section']== 2.0]
    Practica = dataset.loc[dataset['Section']== 3.0]

    return  Gamificacion, TemaConceptual, Practica

def graphs(Gamificacion, TemaConceptual, Practica, Indicator):
    def pie_chart(data, title):
        count_values = data[Indicator].value_counts()
        print(count_values)
        print("Hola")

        fig, ax = plt.subplots(figsize=(5, 5))  
        explode = (0.1, 0)  
        colors = sns.color_palette("pastel")[0:2]

        if len(count_values) == 2:
            values = [count_values.get(1, 0), count_values.get(0, 0)]
            labels = [f"High {Indicator}", f"Low {Indicator}"]
            explode = (0.05, 0.05)
        elif count_values.index[0] == 1:
            values = [count_values[1]]
            labels = [f"High {Indicator}"]
            explode = (0.1,)
        elif count_values.index[0] == 0:
            values = [count_values[0]]
            labels = [f"Low {Indicator}"]
            explode = (0.1,)

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            explode=explode,
            autopct='%1.1f%%',
            startangle=180,
            colors=colors[:len(values)],
            wedgeprops={'edgecolor': 'black'}
        )

        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)

        ax.axis('equal')  # Círculo perfecto
        ax.set_title(title, fontsize=13, fontweight='bold')
        st.pyplot(fig)

    # Generar las gráficas
    pie_chart(Gamificacion, f'Gamification - {Indicator}')
    pie_chart(TemaConceptual, f'Conceptual Topic - {Indicator}')
    pie_chart(Practica, f'Practice - {Indicator}')


st.write("Upload your raw EEG CSV file from Mind Monitor (files can be very large, please be patient)")
uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv", 
        accept_multiple_files=False,
        help="Upload the raw EEG data CSV file from Mind Monitor"
)
    
if uploaded_file is not None:
        # Display file info
        file_details = {"Filename": uploaded_file.name, 
                       "FileType": uploaded_file.type, 
                       "FileSize": f"{uploaded_file.size / (1024 * 1024):.2f} MB"}
        st.write(file_details)
        
        dataset = load_dataset(uploaded_file)
        Indicator = st.radio(
        "What's your favorite movie genre",
        [":rainbow[Alertness]", "***Concentration***", "Fatigue :movie_camera:", "***Motivation***", "Stress :movie_camera:"],
        )

        if Indicator == ":rainbow[Alertness]":
                Indicator = "Alertness"
                rf = load_model(Indicator)
                Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "***Concentration***":
                Indicator = "Concentration"
                rf = load_model(Indicator)
                Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "Fatigue :movie_camera:":
                Indicator = "Fatigue"
                rf = load_model(Indicator)
                Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "***Motivation***":
                Indicator = "Motivation"
                rf = load_model(Indicator)
                Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "Stress :movie_camera:":
                Indicator = "Stress"
                rf = load_model(Indicator)
                Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)
        
