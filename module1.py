import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def plot_fruit_knn(X, y, n_neighbors, weights):

    X_mat = X[['height', 'width']].values

    # Encode fruit names
    le = LabelEncoder()
    y_num = le.fit_transform(y)

    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                         weights=weights)
    clf.fit(X_mat, y_num)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold = ListedColormap(['red', 'lime', 'blue', 'gray'])

    mesh_step_size = 0.01
    plot_symbol_size = 50

    x_min, x_max = X_mat[:,0].min()-1, X_mat[:,0].max()+1
    y_min, y_max = X_mat[:,1].min()-1, X_mat[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    plt.scatter(
        X_mat[:,0],
        X_mat[:,1],
        c=y_num,
        cmap=cmap_bold,
        s=plot_symbol_size,
        edgecolor='black'
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title(f'KNN Classification (k = {n_neighbors})')

    plt.legend(handles=[
        mpatches.Patch(color='red', label='apple'),
        mpatches.Patch(color='lime', label='mandarin'),
        mpatches.Patch(color='blue', label='orange'),
        mpatches.Patch(color='gray', label='lemon')
    ])

    plt.show()


df=pd.read_csv('fruit_data_with_colors.csv')
x=df[['mass','width','height']]
y=df['fruit_name']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
#df.isna().sum()
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
prediction=knn.predict([[15,5.8,10]])


xs=range(1,21)
ys=[]

for i in range( 1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    ys.append(knn.score(x_test,y_test))

plt.scatter(xs,ys)
plt.xlabel('K values of kNN')
plt.ylabel('Accuracy Level %')
plt.xlim(0,20)
plt.xticks([1,5,10,15,20])
plt.ylim(0,1 )
plt.show()
