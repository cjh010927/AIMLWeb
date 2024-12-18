import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## Load the dataset
st.sidebar.header("Upload your CSV file")
uploadedFile = st.sidebar.file_uploader("Upload a SCV file", type=["csv"])

if uploadedFile is not None:
    data = pd.read_csv(uploadedFile)

    st.title("Titanic Survivours Prediction APP")

    ## Show raw dataset
    if "data" in locals() or "data" in globals():
        st.header("Raw dataset")
        isCheck = st.checkbox("Show raw dataset")
        if isCheck:
            st.write(data)

    ## Feature Selection
    # 열의 이름을 제거한 후 리스트로 반환
    importanceFeatures = data.columns.drop(["PassengerId", "Name", "Ticket",
                                            "Cabin", "Survived"]).tolist()
    ## Selecting Features for Prediction
    st.header("Select Features for Prediction")
    #multiselect(설명텍스트, 선택가능한 피처의 전체목록, 기본적으로 선택될 피처)
    selectedFeatures = st.multiselect("Select features to use for predction",
                                      options=importanceFeatures,
                                      default=["Pclass", "Sex", "Age", "Fare", "Embarked"])
    
    # 현재 선택된 피처를 세션 상태로 저장
    if "selectedFeatures" not in st.session_state:
        st.session_state["selectedFeatures"] = selectedFeatures
    # 피처가 바뀌면 모델을 다시 학습하도록 설정
    if selectedFeatures != st.session_state["selectedFeatures"]:
        del st.session_state["gridBestModel"]
        st.session_state["selectedFeatures"] = selectedFeatures

    ## Data Preprocessing
    data = data[selectedFeatures + ['Survived']].dropna()
    data = pd.get_dummies(data, drop_first=True)
    st.write(data)

    ## Splitting the Data
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## Hyperparameter Tuning using GridSearchCV
    st.header(" Hyperparameter Tuning with GridSearchCV")
    paramGrdid = {
        "n_estimators":[50, 100, 150],
        "max_depth":[None, 10, 20],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4]
    }
    # 학습됨 모델을 저장할 수 있는 세션 상태 확인
    if "gridBestModel" not in st.session_state:
        gridSearch = GridSearchCV(RandomForestClassifier(random_state=42),
                                    param_grid=paramGrid, cv=3, n_jobs=-1, verbose=2)
        gridSearch.fit(X_train, y_train)
        st.session_state["gridBestModel"] = gridSearch.best_estimator_
    ## Best Model
    gridBestModel = st.session_state["gridBestModel"]

    ## Making Predictions
    y_pred = gridBestModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the best model: {accuracy * 100:.2f}%")

    ## Feature Importance Analysis
    st.header("Feature Importance")
    featureImportances = np.sort(gridBestModel.feature_imporances_)[::-1]
    featureImportances = featureImportances.reshape(1, -1)
    featureImportances = pd.DataFrame(featureImportances,
                                            columns=X.columns)
    st.write(featureImportances)    
    ## Plotting Feature Importance
    fig, ax = plt.subplots()
    sns.barplot(x = featureImportances.iloc[0,:],
                y = featureImportances.columns, ax = ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    ## User Input for Prediction
st.header("Predict Survival")

userInput = {}
for feature in selectedFeatures:
    if feature == "Pclass":
        userInput[feature] = st.radio(
            "Select Pclass",
            options=[1, 2, 3],
            index=0,
            horizontal=True
        )

    elif feature in ['Fare', 'Age']:
        userInput[feature] = st.slider(
            f"Enter {feature}",
            min_value=float(data[feature].min()),
            max_value=float(data[feature].max()),
            value=float(data[feature].mean())
        )

    elif feature in ['SibSp', 'Parch']:
        userInput[feature] = st.slider(
            f"Enter {feature}",
            min_value=int(data[feature].min()),
            max_value=int(data[feature].max()),
            value=int(data[feature].mean())
        )

    elif feature == "Sex":
        userInput["Sex_male"] = st.selectbox(
            "Select Gender",
            options=['Male', 'Female']
        ) == 'Male'

    elif feature == "Embarked":
        userInput[feature] = st.selectbox(
            "Select Embarkation Port",
            options=['C', 'Q', 'S']
        )
        if userInput[feature] == 'Q':
            userInput["Embarked_Q"] = 1
            userInput["Embarked_S"] = 0
        elif userInput[feature] == 'S':
            userInput["Embarked_Q"] = 0
            userInput["Embarked_S"] = 1
        else:
            userInput["Embarked_Q"] = 0
            userInput["Embarked_S"] = 0

        inputDF = pd.DataFrame([userInput])
        if "Embarked" in inputDF.columns:
            inputDF = inputDF.drop(columns=["Embarked"], axis=1, errors='ignore')

        ## Convert inputDF to numpyArray and ID array to 2D
        inputArray = np.array(inputDF.iloc[0, :])
        inputArray = inputArray.reshape(1, -1)
        st.write(inputArray)

        ## Prediction Button
        if st.button("Predict"):
            prediction = gridBestModel.predict(inputArray)[0]
            if prediction == 1:
                st.success("The passenger is likely to survive!")
            else:
                st.error("The passenger is unlikely to survive")

else:
    st.error("No dataset available. Please upload a CSV file.")