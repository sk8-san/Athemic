
from fastapi import Request, FastAPI
from starlette.responses import FileResponse
from pydantic import BaseModel
app = FastAPI()


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df = pd.read_csv('athlete_events.csv')
df.info()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

a = df[(df['Season']=='Summer')&(df['Year']>=2000)]['Sport'].value_counts()
modernSports = list(a.keys())
len(modernSports)
df1 = df.dropna(subset=['Age', 'Height', 'Weight'])
df1 = df1[df1['Season'] == 'Summer']
df1 = df1.drop(columns=['ID', 'Team', 'Games', 'Season', 'City', 'NOC', 'Event', 'Medal'])
df1 = df1.drop_duplicates()
df1 = df1.drop(columns=['Name'])
le1 = LabelEncoder()
df1['Sex'] = le1.fit_transform(df1['Sex'])
aa = df1['Sport'].value_counts()
allSports1 = list(aa.keys())
a = df1.groupby(['Year'])['Sport'].value_counts()
yearIndex = 0
c = np.zeros(((int)((2020-1896) / 4),len(allSports1)))
for year in range(1896,2020,4):
    if year in a:
        sportIndex = 0
        for sport in allSports1:
            if (sport in a[year]):
                c[yearIndex][sportIndex] = a[year][sport]
            sportIndex = sportIndex + 1
    yearIndex = yearIndex + 1
aa = df1['Sport'].value_counts()

chosenSports = modernSports
sportIndex = 0
for sport in allSports1:
    if sum(c>0)[sportIndex] < 5:
        if sport in chosenSports:
            chosenSports.remove(sport)
    sportIndex = sportIndex + 1
len(chosenSports)
df1 = df1[df1['Sport'].isin(chosenSports)]
df1 = df1.drop(columns=['Year'])
sports = df1['Sport'].unique()
le2 = LabelEncoder()
df1['Sport'] = le2.fit_transform(df1['Sport'])
df1.info()
sns.pairplot(df1)

X = df1.drop(columns=['Sport'])
y = df1['Sport']
scaler = StandardScaler()
scaler.fit(X.values)
scaled_features = scaler.fit_transform(X.values)
df_feat = pd.DataFrame(scaled_features, columns=X.columns)
X1 = df_feat
y1 = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X1.values, y1.values, test_size=0.3)

#Decision tree:
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions_dt = dtree.predict(X_train)
print(classification_report(y_train,predictions_dt))
predictions_dt = dtree.predict(X_test)
print(classification_report(y_test,predictions_dt))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(40, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(chosenSports), activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1)
predictions_nn = model.predict(X_train)
predictions_nn1 = np.argmax(predictions_nn, axis=1)
print(classification_report(y_train,predictions_nn1))
predictions_nn = model.predict(X_test)
predictions_nn1 = np.argmax(predictions_nn, axis=1)
print(classification_report(y_test,predictions_nn1))
counts = np.zeros((len(chosenSports),), dtype=int)
i = 0
for prediction in predictions_nn:
    a = np.argsort(prediction)
    for j in range((len(chosenSports))):
        if y_test[i] == a[-(j+1)]:
            for c in range((len(chosenSports)-j)):
                counts[j+c] = counts[j+c]+1
    i = i+1
counts = counts / i
# accuracy for if target is included in top X suggested sports
print(counts)
plt.plot(counts)
def sport_for_person(sex, age, height, weight, model_type):
    test = np.array([sex, age, height, weight], np.float64)
    test = test.reshape(1, -1)
    test1 = scaler.transform(test)
    toprint = ""
    if model_type == 'svc':
        prediction = svc_model.predict(test1)
        toprint = "[SVC] Suggested sport is: "
        toprint = toprint + sports[prediction]
        print(toprint[0])
    elif model_type == 'dt':
        prediction = dtree.predict(test1)
        toprint = "[DT] Suggested sport is: "
        toprint = toprint + sports[prediction]
        print(toprint[0])
    elif model_type == 'nn':
        prediction = model.predict(test1)
        a = np.argsort(prediction)
        for j in range(10):
            toprint = toprint + sports[a[0][-(j+1)]] + ' (' + (str)((int)(100*prediction[0][a[0][-(j+1)]])) + '%)'
            if (j!=9):
                toprint = toprint + ', '
        return toprint



@app.get("/")
async def read_items():
    return FileResponse('index.html')


@app.post("/predict")
async def getdata(request: Request):
    try:
        body = await request.json()
        for k, v in body.items():
            body[k] = int(v)
        return sport_for_person(body["sex"],body["age"], body["height"], body["weight"],'nn')
    except Exception as e:
        print(e)