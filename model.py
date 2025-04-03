# Importing Data Manipulation Libraries
import pandas as pd
import numpy as np

# Importig the Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Machine Learning Libraries
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importing Filter warning library
import warnings
warnings.filterwarnings('ignore')

# Importing Logging Library
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', force= True,
                    filename= 'Model.log',
                    filemode='w')

URL = 'https://raw.githubusercontent.com/anirudhajohare19/Crop_Recommendation_Model/refs/heads/main/Crop_Recommendation.csv'
df = pd.read_csv(URL)

# Converting Categorical Variables to Numerical Variables ---> Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Crop'] = le.fit_transform(df['Crop'])

# Split the Dataset into X and y
X = df.drop(columns= ['Crop','Rainfall'], axis=1)
y = df['Crop']

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
model = RandomForestClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
   
print('Accuracy:', accuracy_score(y_test, y_pred))