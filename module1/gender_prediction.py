import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
df=pd.read_csv('gender_classification_v7.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=False)

xs=[]
ys=[]
for k in range(1, 375+1):
    xs.append(k)
    print(k)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    a=knn.score(x_test,y_test)
    ys.append(a*100)


plt.plot(xs,ys)

best_k = xs[np.argmax(ys)]
best_score = max(ys)
#
plt.axvline(best_k, color='red', linestyle='--', alpha=0.6)
plt.scatter([best_k], [best_score], color='red', zorder=5)
plt.annotate(f'Best: k={best_k}, {best_score:.1f}%',
             xy=(best_k, best_score),
             xytext=(best_k + 20, best_score - 5),
             fontsize=10, color='red')
#
plt.xlim(1,375)
plt.ylim(0,105)
plt.xlabel('K values')
plt.ylabel('Prediction %')
plt.title('Effect of Knn value in Prediction ')
plt.show()



















print('DONE!')
