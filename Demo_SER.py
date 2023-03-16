import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import FeatureExtraction
import pandas as pd
import keras

# Demo Audio sample
audio = [r'C:\Users\DELL\PycharmProjects\FYP_proj\03-01-05-02-01-01-22.wav']  # Change audio sample directory
df = pd.DataFrame(audio, columns=['Path'])

X = []
for path in df.Path:
    feature = FeatureExtraction.get_features(path)
    X.append(feature)
print("Extraction done")
features_extracted = pd.DataFrame(X)

print(features_extracted.T)

# Expand Dimesion of Input Audio sample to input to model
features_extracted = np.expand_dims(features_extracted, axis=2)

#Load Trained Model
reconstructed_model = keras.models.load_model(r"C:\Users\DELL\PycharmProjects\FYP_proj\SER_model")

def decode(value):
    return 'angry' if value == 0 else ('happy' if value == 1 else 'neutral' if value == 2 else 'sad')


# function to display top two emotions
def use_case_evaluation(prediction):
    ind = (-prediction).argsort()[:2]
    pred = ind[:, :2]
    decoded_string = [decode(val) for val in pred[0]]
    prob_list = [prediction[0][val] for val in pred[0]]

    dict_pred = dict(zip(decoded_string, prob_list))
    return dict_pred, decoded_string
# angry = 0,happy=1 ,neutral=2,,sad = 3

predictions = reconstructed_model.predict(features_extracted)
top2_preds, decoded_string = use_case_evaluation(predictions)
print(top2_preds)

