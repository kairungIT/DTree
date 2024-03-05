import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for classification")
df = pd.read_csv("./data/iris.csv")
st.write(df.head(10))

features=['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
X = df.drop('variety',axis=1)
y = df['variety']

x_train,x_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=200)
ModelDtree = DecisionTreeClassifier()
dtree =ModelDtree.fit(x_train,y_train)

y_predict=dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)  
print(score)  

x_input=[[2.3,4.3,5.0,6.0]] # ใส่ข้อมูลสำหรับการจำแนกข้อมูล
y_predict2=dtree.predict(x_input)
print(y_predict2)

#tree.plot_tree(dtree, feature_names=features)
