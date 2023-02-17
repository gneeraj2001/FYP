import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import librosa

'''Feature Extraction for SER'''


# Extract MFCC Features from audio samples
def mfcc_feature(data, sampling_rate):
    return np.mean(librosa.feature.mfcc(y=data, n_mfcc=40, sr=sampling_rate).T, axis=0)


# Extract Mel Spectrogram Features from audio samples
def mel_spectrogram_feature(data, sampling_rate):
    return np.mean(librosa.feature.melspectrogram(y=data, sr=sampling_rate).T, axis=0)


# Extract Chroma Features from audio samples
def chroma(stft, sampling_rate):
    return np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T, axis=0)


# Extract Tonnetz features from audio samples
def tonnetz(data, sampling_rate):
    return np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sampling_rate).T, axis=0)


# Extract Spectral Contrast from audio samples
def spectral_contrast(data, sampling_rate):
    return np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate).T, axis=0)


# function to extract the required features from the data
def feature_extraction(data, sampling_rate):
    features = np.array([])
    stft = np.abs(librosa.stft(data))

    # MFCC
    mfcc = mfcc_feature(data, sampling_rate)
    features = np.hstack((features, mfcc))

    # Mel Spectrogram
    mel = mel_spectrogram_feature(data, sampling_rate)
    features = np.hstack((features, mel))  # stacking horizontally

    # Chroma
    chrom = chroma(stft, sampling_rate)
    features = np.hstack((features, chrom))

    # tonnetz
    ton = tonnetz(data, sampling_rate)
    features = np.hstack((features, ton))

    # Spectral Contrast
    contr = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate).T, axis=0)
    features = np.hstack((features, contr))

    return features


def get_features(audio):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files
    # OFFSET-START OF THE AUDIO AND DURATION-THE END TIME STAMP OF THE AUDIO
    data, sampling_rate = librosa.load(audio, duration=2, offset=0.5)
    features = feature_extraction(data, sampling_rate)
    result = np.array(features)

    return result




