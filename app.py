# Import libraries
import gradio as gr
import shap
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import dice_ml
from dice_ml import Dice

# Load and train model on Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Initialize DiCE for counterfactuals
data_dice = dice_ml.Data(dataframe=df, continuous_features=X.columns.tolist(), outcome_name='Outcome')
model_dice = dice_ml.Model(model=model, backend='sklearn')
exp = Dice(data_dice, model_dice)

# Gradio interface function
def explain_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                                BMI, DiabetesPedigreeFunction, Age]], columns=X.columns)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    # SHAP values
    shap_values = explainer.shap_values(input_data)
    shap_df = pd.DataFrame({'Feature': X.columns, 'SHAP Value': shap_values[1][0]})
    shap_df = shap_df.sort_values(by='SHAP Value', key=abs, ascending=False)
    shap_plot = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h', title='SHAP Explanation')

    # Counterfactuals with DiCE
    dice_exp = exp.generate_counterfactuals(input_data, total_CFs=1, desired_class="opposite")
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df

    return f"Prediction: {'Diabetic' if prediction else 'Not Diabetic'}\nProbability: {prediction_proba:.2f}", shap_plot, cf_df

# Gradio UI
inputs = [
    gr.Slider(0, 20, label="Pregnancies"),
    gr.Slider(0, 200, label="Glucose"),
    gr.Slider(0, 140, label="BloodPressure"),
    gr.Slider(0, 100, label="SkinThickness"),
    gr.Slider(0, 850, label="Insulin"),
    gr.Slider(10.0, 70.0, label="BMI"),
    gr.Slider(0.0, 2.5, label="DiabetesPedigreeFunction"),
    gr.Slider(20, 80, label="Age")
]

outputs = [
    gr.Textbox(label="Prediction"),
    gr.Plot(label="SHAP Explanation"),
    gr.Dataframe(label="Counterfactual Suggestion")
]

gr.Interface(fn=explain_diabetes, inputs=inputs, outputs=outputs, title="Interactive XAI for Diabetes Prediction").launch()
