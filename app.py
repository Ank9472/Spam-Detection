from flask import Flask, request, render_template
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import os

# Replace with the actual path where vectorizer.pkl is stored
file_path = r"C:\Users\ank94\OneDrive\Desktop\ML spam\vectorizer.pkl"

if os.path.exists(file_path):
    print("File exists!")
    tfidf = pickle.load(open(file_path, 'rb'))
else:
    print(f"File not found at: {file_path}")
# Initialize Flask app
app = Flask(__name__)


# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Required model files are missing. Please ensure 'vectorizer.pkl' and 'model.pkl' are present.")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_sms = request.form.get('message')
        
        if input_sms:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            
            prediction = "Spam" if result == 'spam' else "Not Spam"
            return render_template('index.html', prediction=prediction, message=input_sms)
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)