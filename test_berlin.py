import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder

test_data = pd.read_csv(r'C:\Users\DELL\PycharmProjects\FYP_proj\Datasets\test_berlin.csv')
test_data = test_data.loc[test_data['labels'].isin(['angry', 'sad', 'neutral', 'happy'])]

X_erli = test_data.iloc[:, 1:-1].values
y_erli = test_data.iloc[:, -1].values

# Encode categorical variables
encoder = LabelEncoder()
y_erli = encoder.fit_transform(y_erli)


#Load Model
model = keras.models.load_model(r"C:\Users\DELL\PycharmProjects\FYP_proj\Model_KFOLDS")

X_erli = np.expand_dims(X_erli, axis=2)

loss, acc = model.evaluate(X_erli, y_erli)
print("Accuracy: {:5.2f}%".format(100*acc))
