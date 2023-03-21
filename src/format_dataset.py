from mat4py import loadmat
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from fonction import *


# on importe les données du fichier .mat format de données matlab
data = loadmat("data/new_data_all_7.mat")

data_label = np.array(data["all_data_label"], dtype=object)
data = np.array(data["all_data"], dtype=object)

cols_labels = ["AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax", "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax", "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
               "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax", "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax", "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax", "TypeOfShot"]

# df = pd.DataFrame(data, columns=["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"])
# df_label = pd.DataFrame(data_label, columns=["Stroke Type", "OutSpeed", "OutSpinSpeed", "IncomingSpeed", "IncomingSpinSpeed"])

df = pd.DataFrame(columns=cols_labels)


for N_sujet in range(len(data)):
    for N_coup in range(len(data[N_sujet][0])):
        donnee = np.array(data[N_sujet][0][N_coup][0])
        donne_label = np.array(data_label[N_sujet][0][N_coup])
        row = list()
        for N_cols in range(6):
            donnee[:, N_cols] = (donnee[:, N_cols]-np.mean(donnee[:, N_cols]))/(np.std(donnee[:, N_cols]))
            mean = np.mean(donnee[:, N_cols])
            sd = np.std(donnee[:, N_cols])
            skewness = skew(donnee[:, N_cols])
            kurtosisness = kurtosis(donnee[:, N_cols])
            minimum = np.min(donnee[:, N_cols])
            maximum = np.max(donnee[:, N_cols])
            row.append(mean)
            row.append(sd)
            row.append(skewness)
            row.append(kurtosisness)
            row.append(minimum)
            row.append(maximum)
        row.append(donne_label[0][0])
        df.loc[len(df)] = row
print(df)
df.to_csv("normalize_dataset2")