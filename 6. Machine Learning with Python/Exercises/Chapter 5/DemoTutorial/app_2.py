import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import gradio as gr

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Function to make predictions with probability scores
def predict_iris_prob(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prob_scores = rf_classifier.predict_proba(features)[0]
    return {"Setosa": prob_scores[0], "Versicolor": prob_scores[1], "Virginica": prob_scores[2]}

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_iris_prob,
    inputs=["text", "text", "text", "text"],
    outputs="label",
    title="Iris Flower Classification with Probability Scores",
    description="Enter the values of sepal length, sepal width, petal length, and petal width.",
    examples=[["5.1", "3.5", "1.4", "0.2"],
              ["6.4", "3.2", "4.5", "1.5"]]
)

# Launch the Gradio interface
iface.launch()