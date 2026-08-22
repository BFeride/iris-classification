import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

fayl = pd.read_csv('Iris.csv')

olculer = fayl[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
adlar = fayl['Species']

oyrenolcu, testolcu, oyrenad, testad = train_test_split(olculer, adlar, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(oyrenolcu, oyrenad)

neticeler = model.predict(testolcu)

faiz = accuracy_score(testad, neticeler)
print(f"Deqiqlik: {faiz * 100}%")