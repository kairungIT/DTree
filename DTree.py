import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for classification")
df = pd.read_csv("/data/iris.csv")
st.write(df.head(10))

features=['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
X = df.drop('variety',axis=1)
y = df['variety']

x_train,x_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=200)
ModelDtree = DecisionTreeClassifier()
dtree =ModelDtree.fit(x_train,y_train)
tree.plot_tree(dtree, feature_names=features)
