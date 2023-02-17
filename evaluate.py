import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import FeatureExtraction
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Function to split the data into train and test set
def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

    return X_train, X_test, y_train, y_test


# Dataset Loading
data = pd.read_csv(r'C:\Users\DELL\PycharmProjects\FYP_proj\Datasets\dataset7.csv')
emotions_filter = ['angry', 'sad', 'neutral', 'happy']

# 376+376+376+188 => Total Number of samples in new dataset (reduce size of dataset)
data = data.loc[data['labels'].isin(['angry', 'sad', 'neutral', 'happy'])]
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = split(X,y)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)



def evaluate_model(model):
    loss, acc = model.evaluate(X_test, y_test)
    print("Accuracy: {:5.2f}%".format(100 * acc))
    return loss,acc

reconstructed_model = keras.models.load_model(r"C:\Users\DELL\PycharmProjects\FYP_proj\Model_KFOLDS")


loss,acc = evaluate_model(reconstructed_model)