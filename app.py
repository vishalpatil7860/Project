from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the preprocessed model and the preprocessor
model = load_model('recommendation_model')
preprocessor = joblib.load('preprocessor.joblib')

# Load your dataset to get the list of unique items
data = pd.read_csv('data.csv', encoding='ISO-8859-1')
unique_items = data['Description'].unique().tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_product = None  # Initialize selected_product to None outside of the POST block
    
    if request.method == 'POST':
        selected_product = request.form.get('product')
        preprocessed_input = preprocess_input(selected_product)
        predictions = model.predict(preprocessed_input)
        recommendations = get_recommendations(predictions, unique_items)

    # Ensure selected_product is passed to the template regardless of whether it's a POST request
    return render_template('index.html', products=unique_items, selected_product=selected_product, recommendations=recommendations)


def preprocess_input(product):
    # Preprocess the input similar to how the training data was preprocessed
    input_data = pd.DataFrame([[product, 0, 0, 'United Kingdom', 0]], columns=['Description', 'Quantity', 'UnitPrice', 'Country', 'TotalPrice'])
    transformed = preprocessor.transform(input_data)
    return transformed.toarray()

def get_recommendations(predictions, unique_items):
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    return [unique_items[i] for i in top_indices]

if __name__ == '__main__':
    app.run(debug=True)
