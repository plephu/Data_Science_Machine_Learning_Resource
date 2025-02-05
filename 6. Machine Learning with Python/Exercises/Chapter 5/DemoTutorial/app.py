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

# Set minimum and maximum values for sliders
slider_min = X_train.min(axis=0)
slider_max = X_train.max(axis=0)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Function to make predictions with probability scores and feature importance
def predict_iris_prob(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prob_scores = rf_classifier.predict_proba(features)[0]
    feature_importance = rf_classifier.feature_importances_
    feature_importance = {data.feature_names[i]: importance for i, importance in enumerate(feature_importance)}
    return {"Setosa": prob_scores[0], "Versicolor": prob_scores[1], "Virginica": prob_scores[2]}
    

# Create the Gradio interface with sliders for inputs
iface = gr.Interface(
    fn=predict_iris_prob,
    inputs=[
        gr.inputs.Slider(minimum=slider_min[0], maximum=slider_max[0], step=0.1, default=slider_min[0], label="Sepal Length"),
        gr.inputs.Slider(minimum=slider_min[1], maximum=slider_max[1], step=0.1, default=slider_min[1], label="Sepal Width"),
        gr.inputs.Slider(minimum=slider_min[2], maximum=slider_max[2], step=0.1, default=slider_min[2], label="Petal Length"),
        gr.inputs.Slider(minimum=slider_min[3], maximum=slider_max[3], step=0.1, default=slider_min[3], label="Petal Width"),
    ],
    outputs=["label"],
    title="Iris Flower Classification with Probability Scores and Feature Importance",
    description="Adjust the sliders to set sepal length, sepal width, petal length, and petal width.",
    examples=[["5.1", "3.5", "1.4", "0.2"],
              ["6.4", "3.2", "4.5", "1.5"]],
               live=True
)

# Print model's accuracy on the test set
accuracy = rf_classifier.score(X_test, y_test)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")

# Launch the Gradio interface
iface.launch(share=True)
