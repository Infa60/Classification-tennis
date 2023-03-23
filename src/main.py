from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fonction import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC



cols_labels = ["AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax", "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax", "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
               "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax", "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax", "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax"]

frequence = 60
intervalle = 30 #Intervalle doit pouvoir s'adapter pour le premier et le dernier coup si l'on coupe ou lance trop tot l'enregistrement mais rester à 30 pour le reste des essais
pourcentage_max = 40

new_shot_data = pd.read_csv("data/092212.csv", skiprows=range(0,11), usecols=range(2,8))
new_shot_data.columns = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]

#Filtrage des données
#new_shot_data["AccX"]=LP_filter(new_shot_data["AccX"],6,frequence) 
#new_shot_data["AccY"]=LP_filter(new_shot_data["AccY"],6,frequence) 
#new_shot_data["AccZ"]=LP_filter(new_shot_data["AccZ"],6,frequence) 
#new_shot_data["GyrX"]=LP_filter(new_shot_data["GyrX"],6,frequence) 
#new_shot_data["GyrY"]=LP_filter(new_shot_data["GyrY"],6,frequence) 
#new_shot_data["GyrZ"]=LP_filter(new_shot_data["GyrZ"],6,frequence) 

maxX = np.max(new_shot_data["AccX"])
seuil_max = maxX * pourcentage_max / 100
AccX_peaks = find_peaks(new_shot_data["AccX"],height=seuil_max, distance=40)

print(AccX_peaks)

if (AccX_peaks[0][0]) <= intervalle :
    intervalle=(AccX_peaks[0][0])

if (len(new_shot_data)-AccX_peaks[0][-1]) <= intervalle :
    intervalle=(len(new_shot_data)-AccX_peaks[0][-1])

plt.plot(new_shot_data["AccX"])
plt.axhline(maxX, color='red', linestyle='--', label='Maximum')
plt.axhline(seuil_max, color='green', linestyle='-.', label='Seuil de detection')
plt.legend()


#plt.plot(new_shot_data["AccY"])
#plt.plot(new_shot_data["AccZ"])
plt.show()
peaks = AccX_peaks[0]



NB_shots = len(AccX_peaks[0])

to_predict_shot = pd.DataFrame(columns=cols_labels)

for i in peaks:
    data_new_shot = new_shot_data[i-intervalle:i+intervalle]
    row = list()
    for j in range(6):
        mean = np.mean(data_new_shot.iloc[:, j])
        sd = np.std(data_new_shot.iloc[:, j])
        skewness = skew(data_new_shot.iloc[:, j])
        kurtosisness = kurtosis(data_new_shot.iloc[:, j])
        minimum = np.min(data_new_shot.iloc[:, j])
        maximum = np.max(data_new_shot.iloc[:, j])
        row.append(mean)
        row.append(sd)
        row.append(skewness)
        row.append(kurtosisness)
        row.append(minimum)
        row.append(maximum)

    to_predict_shot.loc[len(to_predict_shot)] = row

print(to_predict_shot)

data = pd.read_csv("dataset/training_dataset")
data.drop(457, axis=0, inplace=True)
data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

X = data.loc[:, data.columns != "TypeOfShot"]
Y = data["TypeOfShot"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=250,criterion="entropy",max_depth=12,min_samples_split=4,min_samples_leaf=1,max_features=8,bootstrap=False)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

typeofshot = clf.predict(to_predict_shot)
print(typeofshot)

print("Accuracy : ", accuracy_score(Y_test, y_pred))
print("Precision : ", precision_score(Y_test, y_pred, average='weighted', zero_division=1))
print("___________")


clf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
clf.fit(X_train, Y_train)
y_pred3 = clf.predict(X_test)

typeofshot3 = clf.predict(to_predict_shot)
print(typeofshot3)

print("Accuracy : ", accuracy_score(Y_test, y_pred3))
print("Precision : ", precision_score(Y_test, y_pred3, average='weighted', zero_division=1))
print("___________")



