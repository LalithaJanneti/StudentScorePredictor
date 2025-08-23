import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#load Dataset

@st.cache_data
def load_data():
    df=pd.read_csv("StudentsPerformance.csv")
    df=pd.get_dummies(df,drop_first=True)
    return df

df=load_data()

st.title("📊 Student Performance Prediction App")
st.write("This app predicts **Performance Index** based on the features like math score, writing score and reading score")

df["Performance Index"]=df[["math score","reading score","writing score"]].mean(axis=1)

X=df.drop("Performance Index",axis=1)
y=df["Performance Index"]

#train test split method
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model training
model=LinearRegression()
model.fit(X_train,y_train)

#predictions
y_pred = model.predict(X_test)

#metrics operations
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

st.subheader("📈 Model Performance")
st.write(f"**MSE:**{mse:.4f}")
st.write(f"**MAE:**{mae:.4f}")
st.write(f"**R2 score:**{r2:.4f}")

st.subheader("📝 Enter Student Details")

reading = st.slider("Reading Score",0,100,50)
writing = st.slider("Writing Score",0,100,50)
gender = st.selectbox("Gender",["female","male"])
lunch = st.selectbox("Lunch",["Standard","free/reduced"])
test_prep = st.selectbox("test Preparation Course",["none","completed"])
parent_edu = st.selectbox("Parental Eduction",["some high school","high school","some college","bachelor's degree","master's degree"])

#Encode inputs same as training 
input_dict = {
    "reading score":reading,
    "writing score":writing,
    "gender_male":1 if gender =="male" else 0,
    "lunch_standard":1 if lunch =="standard" else 0,
    "test preparation course_none": 1 if test_prep == "none" else 0,
    "parental level of education_bachelor's degree": 1 if parent_edu=="bachelor's degree" else 0,
    "parental level of education_high school":1 if parent_edu == "high school" else 0,
    "parental level of education_master's degree":1 if parent_edu == "master's degree" else 0,
    "parental level of education_some college": 1 if parent_edu == "some college" else 0,
    "parental level of education_some high school":1 if parent_edu == "high school" else 0,

}

#ensure all columns present
input_df=pd.DataFrame([input_dict])
for col in X.columns:
    if col not in input_df.columns:
        input_df[col]=0
input_df = input_df[X.columns]

#predictions
if st.button("Predict Performance Index"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted student Performance Index: **{prediction:.2f}**")
