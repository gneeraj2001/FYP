import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


erli_data = pd.read_csv(r'C:\Users\DELL\PycharmProjects\FYP_proj\Datasets\test_berlin.csv')
erli_data = erli_data.loc[erli_data['labels'].isin(['angry', 'sad', 'neutral', 'happy'])]

X_erli = erli_data.iloc[:, 1:-1].values
y_erli = erli_data.iloc[:, -1].values

# Encode categorical variables
encoder = LabelEncoder()
y_erli = encoder.fit_transform(y_erli)


#Load Model
model = keras.models.load_model(r"C:\Users\DELL\PycharmProjects\FYP_proj\Model_KFOLDS")
X_erli = np.expand_dims(X_erli, axis=2)

loss, acc = model.evaluate(X_erli, y_erli)
print("Accuracy: {:5.2f}%".format(100*acc))

predict = model.predict(X_erli)


predictions = model.predict(X_erli).argmax(axis=1)

print(predictions)
print(y_erli)

print(accuracy_score(y_true=y_erli,y_pred=predictions))
print(classification_report(y_erli,predictions))
# creating a confusion matrix
print(confusion_matrix(y_erli, predictions) )
