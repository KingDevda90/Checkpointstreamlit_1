import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
from ydata_profiling import ProfileReport

# Utilisation de Streamlit pour afficher du texte
# st.title("Ma première application Streamlit")
# st.write("Bonjour, c'est une application simple en Streamlit.")
# ------------------Exploration des données-----------------------#
data = pd.read_csv('/Users/mac/Downloads/Expresso_churn_dataset.csv')
data.head()
data.describe()
data.info()
data.isna().sum()
data.shape
# --------------------Rapport de profilage-------------------#
Profil = ProfileReport(data, title='Pandas Profiling')
Profil.to_file("your_report_name.html")
#---------------------Gestion des valeures manquantes-------------#
# Remplacer les valeurs manquantes par la mode pour les colonnes spécifiques
data[['REGION']] = data[['REGION']].fillna(data[['REGION']].mode().iloc[0])
data[['TOP_PACK']] = data[['TOP_PACK']].fillna(data[['TOP_PACK']].mode().iloc[0])
# Remplacer les valeurs manquantes par la moyenne pour les colonnes spécifiques
data[['MONTANT', 'FREQUENCE_RECH', 'REVENUE','ARPU_SEGMENT','FREQUENCE','DATA_VOLUME','ON_NET','ORANGE','TIGO','ZONE1','ZONE2','FREQ_TOP_PACK']] = data[['MONTANT', 'FREQUENCE_RECH', 'REVENUE','ARPU_SEGMENT','FREQUENCE','DATA_VOLUME','ON_NET','ORANGE','TIGO','ZONE1','ZONE2','FREQ_TOP_PACK']].fillna(data[['MONTANT', 'FREQUENCE_RECH', 'REVENUE','ARPU_SEGMENT','FREQUENCE','DATA_VOLUME','ON_NET','ORANGE','TIGO','ZONE1','ZONE2','FREQ_TOP_PACK']].mean())
#----------------------------------
df = data.drop(['user_id','MRG'], axis=1)
#----------------------Encodade des valeures----------------#
df = pd.get_dummies(df, columns=['REGION', 'TOP_PACK'], prefix=['REGION', 'TOP_PACK'])
#--------------------------Construction du modele------------------#
label_encoder = LabelEncoder()
df['TENURE'] = label_encoder.fit_transform(df['TENURE'])
X = df.drop('TENURE', axis=1)
y = df['TENURE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialiser le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Entraîner le modèle
model.fit(X_train, y_train)
# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy}")
#------------------Construction de l'application---------------#

# Interface utilisateur Streamlit
st.title("Application de prédiction")

# Ajouter des champs de saisie pour les caractéristiques du modèle
feature1 = st.number_input("Saisir la caractéristique 1")
feature2 = st.number_input("Saisir la caractéristique 2")
# Ajoutez d'autres champs de saisie pour les fonctionnalités

# Faire une prédiction sur les nouvelles entrées
if st.button("Prédire"):
    new_data = [[feature1, feature2]]  # Adapter aux caractéristiques spécifiques
    prediction = model.predict(new_data)
    st.success(f"La prédiction est {prediction}")