# ExplainableAI
# Interactive Explanations for Healthcare XAI

This project shows interactive explainability techniques for a healthcare use case (diabetes prediction). It includes:
- SHAP explanations
- Counterfactual examples (DiCE)
- Gradio-based interactive UI

## Features

- Input patient data and receive model prediction
- Visualize SHAP values interactively using Plotly
- Explore "what-if" counterfactual scenarios via DiCE
- Intuitive Gradio web interface

## Installation

1. Clone the repository:
git clone https://github.com/dorinerooseboom/ExplainableAI.git 
cd ExplainableAI

2. Install requirements
pip install -r requirements.txt

Sample requirements.txt:
gradio
scikit-learn
pandas
shap
plotly
dice-ml

## Usage
Run the app with:
python app.py

Then, open the Gradio interface in your browser to:
Enter patient health metrics
View diabetes prediction
Explore SHAP explanations and counterfactuals

## Files
app.py – Main script

## Dataset
Pima Indians Diabetes Dataset

## Authors
Final Project — Explainable AI
Dorine Rooseboom
