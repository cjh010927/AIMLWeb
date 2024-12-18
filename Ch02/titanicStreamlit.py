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

    st.title("Titanic Survivors Prediction APP")

    # Show raw dataset
    st.header("Raw Dataset")
    isCheck = st.checkbox("Show raw dataset")
    if isCheck:
        st.write(data)

    # Feature Selection
    importanceFeatures = data.columns.drop(["PassengerId", "Name", "Ticket", "Cabin", "Survived"]).tolist()

    st.header("Select Features for Prediction")
    selectedFeatures = st.multiselect(
        "Select features to use for prediction",
        options=importanceFeatures,
        default=["Pclass", "Sex", "Age", "Fare", "Embarked"]
    )

    # Update session state for selected features
    if "selectedFeatures" not in st.session_state:
        st.session_state["selectedFeatures"] = selectedFeatures

    if selectedFeatures != st.session_state["selectedFeatures"]:
        st.session_state["selectedFeatures"] = selectedFeatures
        if "gridBestModel" in st.session_state:
            del st.session_state["gridBestModel"]

    # Data Preprocessing
    data = data[selectedFeatures + ["Survived"]].dropna()
    data = pd.get_dummies(data, drop_first=True)
    st.write("Processed Dataset:")
    st.write(data)

    # Splitting the Data
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning using GridSearchCV
    st.header("Hyperparameter Tuning with GridSearchCV")
    paramGrid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    if "gridBestModel" not in st.session_state:
        gridSearch = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=paramGrid, cv=3, n_jobs=-1, verbose=2
        )
        gridSearch.fit(X_train, y_train)
        st.session_state["gridBestModel"] = gridSearch.best_estimator_

    gridBestModel = st.session_state["gridBestModel"]

    # Making Predictions
    y_pred = gridBestModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the best model: {accuracy * 100:.2f}%")

    # Feature Importance Analysis
    st.header("Feature Importance")
    featureImportances = gridBestModel.feature_importances_
    featureImportancesDF = pd.DataFrame({
        "Feature": X.columns,
        "Importance": featureImportances
    }).sort_values(by="Importance", ascending=False)
    st.write(featureImportancesDF)

    # Plotting Feature Importance
    fig, ax = plt.subplots()
    sns.barplot(x=featureImportancesDF["Importance"], y=featureImportancesDF["Feature"], ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # User Input for Prediction
    st.header("Predict Survival")
    userInput = {}
    for feature in selectedFeatures:
        if feature == "Pclass":
            userInput[feature] = st.radio("Select Pclass", options=[1, 2, 3], index=0, horizontal=True)
        elif feature in ["Fare", "Age"]:
            userInput[feature] = st.slider(
                f"Enter {feature}",
                min_value=float(data[feature].min()),
                max_value=float(data[feature].max()),
                value=float(data[feature].mean())
            )
        elif feature in ["SibSp", "Parch"]:
            userInput[feature] = st.slider(
                f"Enter {feature}",
                min_value=int(data[feature].min()),
                max_value=int(data[feature].max()),
                value=int(data[feature].mean())
            )
        elif feature == "Sex":
            userInput["Sex_male"] = st.selectbox("Select Gender", options=["Male", "Female"]) == "Male"
        elif feature == "Embarked":
            embark = st.selectbox("Select Embarkation Port", options=["C", "Q", "S"])
            userInput["Embarked_Q"] = 1 if embark == "Q" else 0
            userInput["Embarked_S"] = 1 if embark == "S" else 0

    inputDF = pd.DataFrame([userInput])
    inputDF = inputDF.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict"):
        prediction = gridBestModel.predict(inputDF)[0]
        if prediction == 1:
            st.success("The passenger is likely to survive!")
        else:
            st.error("The passenger is unlikely to survive.")
else:
    st.error("No dataset available. Please upload a CSV file.")