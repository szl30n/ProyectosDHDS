import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import normalize


###   General ###########################################################################################
#todo natriz de confusions
def _print_matriz_confusion():
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.rcParams['font.size'] = 10
    sns.heatmap(data.iloc[:, :].corr(), vmin = -1, vmax = 1, center = 0, cmap = "YlGnBu", annot = True)
    
def _print_matriz_correlacion(dflocal):
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.rcParams['font.size'] = 10
    sns.heatmap(dflocal.iloc[:, :].corr(), vmin = -1, vmax = 1, center = 0, cmap = "YlGnBu", annot = True)
       
def _get_info(dflocal,h=3):
    print(dflocal.head(h))
    print(dflocal.shape)

def _summary(dflocal):
    return pd.DataFrame({'notnull': dflocal.apply(lambda x: x.notnull().sum()),
                         'dtype': dflocal.apply(lambda x: x.dtype),
                         'unique': dflocal.apply(lambda x: sorted(x.unique()) if len(x.unique()) <= 10 else '> 10')})

    
    
###   regrsion lineal simple ###################################################################################
def _get_rls(X,y,columna):
    # Como estamos trabajando con observaciones ordenadas en el tiempo, ponemos
    # shuffle=False para evitar data leakage    shuffle=False
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 1, shuffle = False)
    lm = LinearRegression() # Fiteamos el modelo sobre los vectores X e y.
    model = lm.fit(Xtrain, ytrain)
    print(f'''    Coef\t{model.coef_}''')
    print(f'''    intercept\t{model.intercept_}''')
    test_sklearn = np.array(Xtest).reshape(-1,1)
    ypred = model.predict(test_sklearn)
    
    print(f'''    MAE\t{mean_absolute_error(ytest, ypred).round(2)}
    MSE\t{mean_squared_error(ytest, ypred).round(2)}
    RMSE\t{np.sqrt(mean_squared_error(ytest, ypred)).round(2)}
    R2\t{r2_score(ytest, ypred).round(2)}
    ''')

    sns.regplot(data = data, x = columna, y = 'precio_por_m2', ci = 95, scatter_kws = {"color": "blue", 's': 10}, line_kws = {"color": "red"})
    #return mean_absolute_error(y_test, y_pred)

 
###   regrsion lineal multiple ###################################################################################
def _rlm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
        
    n = len(y_train)
    p = X_train.shape[1]
    R2 = r2_score(y_test, y_pred)
    R2_ajustado = 1 - (1 - R2)*((n-1)/(n - p - 1))
    
    print(f'''    
    Intercepto\t{linreg.intercept_.round(4)}
    MAE\t\t{mean_absolute_error(y_test, y_pred).round(4)}
    MSE\t\t{mean_squared_error(y_test, y_pred).round(4)}
    RMSE\t{np.sqrt(mean_squared_error(y_test, y_pred)).round(4)}
    R2\t\t{(R2).round(4)}
    R2ADJ\t{(R2_ajustado).round(4)}
    ''')

    plt.plot(y,y, '-.',c='grey')
    plt.scatter(y_pred, y_test, s=30, c='r', marker='+', zorder=10)
    plt.xlabel("Valores predichos")
    plt.ylabel("Valores reales")
    plt.title('Predicción de precio por m$^2$ utilizando RLM')
    plt.show()