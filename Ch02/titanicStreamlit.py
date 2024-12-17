import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
st.sidebar.header("Upload your CSV file")
uploadedFile = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploadedFile is not None:
    data = pd.read_csv(uploadedFile)

    st.title("Titanic Survivors Prediction App")

    # Show raw dataset
    st.header("Raw dataset")
    isCheck = st.checkbox("Show raw dataset")
    if isCheck:
        st.write(data)

    # Feature Selection
    importanceFeatures = data.columns.drop(["PassengerId", "Name", "Ticket", "Cabin", "Survived"]).tolist()

    st.header("Select Features for Prediction")
    selectedFeatures = st.multiselect("Select features to use for prediction",
                                      options=importanceFeatures,
                                      default=["Pclass", "Sex", "Age", "Fare", "Embarked"])

    # Data Preprocessing
    data = data[selectedFeatures + ['Survived']].dropna()
    data = pd.get_dummies(data, drop_first=True)

    # Splitting the Data
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning
    st.header("Hyperparameter Tuning with GridSearchCV")
    paramGrid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    if "gridBestModel" not in st.session_state:
        gridSearch = GridSearchCV(RandomForestClassifier(random_state=42),
                                  param_grid=paramGrid, cv=3, n_jobs=-1, verbose=2)
        gridSearch.fit(X_train, y_train)
        st.session_state["gridBestModel"] = gridSearch.best_estimator_

    gridBestModel = st.session_state["gridBestModel"]

    # Model Accuracy
    y_pred = gridBestModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the best model: {accuracy * 100:.2f}%")

    # Feature Importance
    st.header("Feature Importance")
    featureImportances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': gridBestModel.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.write(featureImportances)

    # Plot Feature Importance
    fig, ax = plt.subplots()
    sns.barplot(x=featureImportances['Importance'], y=featureImportances['Feature'], ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # User Input for Prediction
    st.header("Predict Survival")
    userInput = {}
    for feature in selectedFeatures:
        if feature == "Pclass":
            userInput[feature] = st.radio("Select Pclass", options=[1, 2, 3], index=0, horizontal=True)
        elif feature in ['Fare', 'Age']:
            userInput[feature] = st.slider(f"Enter {feature}", float(data[feature].min()), float(data[feature].max()))
        elif feature == "Sex":
            userInput["Sex_male"] = st.selectbox("Select Gender", ['Male', 'Female']) == 'Male'
        elif feature == "Embarked":
            embarked = st.selectbox("Select Embarkation Port", ['C', 'Q', 'S'])
            userInput["Embarked_Q"] = 1 if embarked == 'Q' else 0
            userInput["Embarked_S"] = 1 if embarked == 'S' else 0

    inputDF = pd.DataFrame([userInput])
    st.write("User Input:", inputDF)

    if st.button("Predict"):
        prediction = gridBestModel.predict(inputDF)[0]
        st.success("The passenger is likely to survive!" if prediction == 1 else "The passenger is unlikely to survive.")
else:
    st.error("No dataset available. Please upload a CSV file.")
