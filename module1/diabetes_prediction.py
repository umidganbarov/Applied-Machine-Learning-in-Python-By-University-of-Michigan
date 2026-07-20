import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('diabetes.csv')
df.Outcome=np.where(df.Outcome==1,'Cancer','NOT-Cancer')
x=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=df.Outcome
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
ys=[]
xs=[x for x in range(1, 576+1)]
for k in range(1,576+1 ):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    ys.append(knn.score(x_test,y_test)*100)

plt.plot(xs,ys)
plt.grid(True, alpha=0.3)
best_k=xs[np.argmax(ys)]
best_score= max(ys)
plt.axvline(best_k,color='red',linestyle='--',alpha=1,label=f'Best k={best_k} ({best_score:.1f} %)')
plt.legend()
plt.xlabel('K values')
plt.ylabel('Prediction %')
plt.ylim(0,105)
plt.xlim(0,577)
plt.show()
