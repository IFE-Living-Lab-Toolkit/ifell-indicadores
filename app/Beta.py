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

def starting():
        Indicator = st.radio(
            "What's your favorite movie genre",
            [":rainbow[Alertness]", "***Concentration***", "Fatigue :movie_camera:", "***Motivation***", "Stress :movie_camera:"],
            )

        if Indicator == ":rainbow[Alertness]":
                Indicator = "Alertness"
                rf = load_model(Indicator)
                LineaBase, Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "***Concentration***":
                Indicator = "Concentration"
                rf = load_model(Indicator)
                LineaBase, Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "Fatigue :movie_camera:":
                Indicator = "Fatigue"
                rf = load_model(Indicator)
                LineaBase, Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "***Motivation***":
                Indicator = "Motivation"
                rf = load_model(Indicator)
                LineaBase, Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)

        elif Indicator == "Stress :movie_camera:":
                Indicator = "Stress"
                rf = load_model(Indicator)
                LineaBase, Gamificacion, TemaConceptual, Practica = prediction(rf, dataset, Indicator)
                graphs(Gamificacion, TemaConceptual, Practica, Indicator)
        
def load_dataset(data):
    #dataset = pd.read_csv("C:/Users/ifeex/OneDrive/Documentos/Modelos Predictivos/Modelos Predictivos/Predicción/[FINAL]EEG_Dataset.csv", header=0)

    dataset['RazonTP9(B/T)'] = np.where(dataset['Theta_TP9'] != 0, dataset['Beta_TP9'] / dataset['Theta_TP9'], 0)
    dataset['RazonAF7(B/T)'] = np.where(dataset['Theta_AF7'] != 0, dataset['Beta_AF7'] / dataset['Theta_AF7'], 0)
    dataset['RazonAF8(B/T)'] = np.where(dataset['Theta_AF8'] != 0, dataset['Beta_AF8'] / dataset['Theta_AF8'], 0)
    dataset['RazonTP10(B/T)'] = np.where(dataset['Theta_TP10'] != 0, dataset['Beta_TP10'] / dataset['Theta_TP10'], 0)
    dataset['X_Razon(B/T)'] = np.where(dataset[['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10']].mean(axis=1) != 0,
                                    dataset[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].mean(axis=1) / 
                                    dataset[['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10']].mean(axis=1), 0)
    dataset['S_Razon(B/T)'] = np.where(dataset[['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10']].std(axis=1) != 0,
                                    dataset[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].std(axis=1) / 
                                    dataset[['Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10']].std(axis=1), 0)
    

    dataset = dataset.dropna()


    return dataset

def load_model(Indicator):
        if (Indicator == "Alertness"):
                rf = joblib.load('C:/Users/SRV-ife.livinglab/Documents/Modelos Predictivos/Repo_Modelos/ifell-indicadores/Modulos/' + Indicator + '/RF91-EEG-' + Indicator + '.joblib')
                return rf
        elif (Indicator == "Concentration"):
                rf = joblib.load('C:/Users/SRV-ife.livinglab/Documents/Modelos Predictivos/Repo_Modelos/ifell-indicadores/Modulos/' + Indicator + '/RF99-EEG-' + Indicator + '.joblib')
                return rf
        elif (Indicator == "Fatigue"):
               rf = joblib.load('C:/Users/SRV-ife.livinglab/Documents/Modelos Predictivos/Repo_Modelos/ifell-indicadores/Modulos/' + Indicator + '/RF99-EEG-' + Indicator + '.joblib')
               return rf
        elif (Indicator == "Motivation"):
               rf = joblib.load('C:/Users/SRV-ife.livinglab/Documents/Modelos Predictivos/Repo_Modelos/ifell-indicadores/Modulos/' + Indicator + '/RF99-EEG-' + Indicator + '.joblib')
               return rf
        elif (Indicator == "Stress"):
               rf = joblib.load('C:/Users/SRV-ife.livinglab/Documents/Modelos Predictivos/Repo_Modelos/ifell-indicadores/Modulos/' + Indicator + '/RF99-EEG-' + Indicator + '.joblib')
               return rf

def prediction(rf, dataset, Indicator):
    X = dataset[['Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8',
            'Beta_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Gamma_TP9','Gamma_AF7',
            'Gamma_AF8','Gamma_TP10','Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
            'RazonTP9(B/T)','RazonAF7(B/T)','RazonAF8(B/T)','RazonTP10(B/T)',
            'X_Razon(B/T)','S_Razon(B/T)']]
    
    y_pred = rf.predict(X)
    dataset[Indicator] = y_pred

    LineaBase = dataset.loc[(dataset['TimeStamp'] >= "2025-03-11 11:15") & (dataset['TimeStamp'] <="2025-03-11 11:18:59")]
    Gamificacion = dataset.loc[(dataset['TimeStamp'] >= "2025-03-11 11:24") & (dataset['TimeStamp'] <="2025-03-11 11:31:59")]
    TemaConceptual = dataset.loc[(dataset['TimeStamp'] >= "2025-03-11 11:32") & (dataset['TimeStamp'] <="2025-03-11 11:52:59")]
    Practica = dataset.loc[(dataset['TimeStamp'] >= "2025-03-11 11:53") & (dataset['TimeStamp'] <="2025-03-11 12:33:59")]

    return LineaBase, Gamificacion, TemaConceptual, Practica


def graphs(Gamificacion, TemaConceptual, Practica, Indicator):
    def pie_chart(data, title):
        count_values = data[Indicator].value_counts()
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
        else:
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


    c1,c2,c3 = st.columns(3)
    # Generar las gráficas
    with c1:

        pie_chart(Gamificacion, f'Gamification - {Indicator}')
    with c2:

        pie_chart(TemaConceptual, f'Conceptual Topic - {Indicator}')
    with c3:

        pie_chart(Practica, f'Practice - {Indicator}')
    
    c1,c2,c3 = st.columns(3)


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

        dataset = pd.read_csv(uploaded_file)
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset = dataset.dropna()
        dataset['TimeStamp'] = pd.to_datetime(dataset['TimeStamp'])

        if ((len(dataset.columns) == 7) & (dataset.columns[6].lower() == "temperature")):
            opcion = 1
            dataset = load_dataset(uploaded_file)

        elif (len(dataset.columns) == 28 & dataset.columns[4] == 'Delta_TP9'):
            opcion = 2
            dataset = load_dataset_embrace(uploaded_file)
        else:
            st.write("No es lo buscado, intentelo de nuevo")
       


