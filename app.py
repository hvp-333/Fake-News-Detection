from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

def preprocess_text(text):
    # Convert text to lowercase
    text = str(text).lower()
    # Remove non-alphanumeric characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load the traditional ML models
rfc_model = joblib.load('RFC_Model.pkl')
mlp_model = joblib.load('MLP_Model.pkl')
lr_model = joblib.load('LR_Model.pkl')
dt_model = joblib.load('DT_Model.pkl')
gbc_model = joblib.load('GBC_Model.pkl')
mnb_model = joblib.load('MNB_Model.pkl')

# Load the BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForSequenceClassification.from_pretrained(model_name)
bert_model.load_state_dict(torch.load('bert_model.pth'))
bert_model.eval()

def predict_with_bert(news_text):
    inputs = tokenizer(news_text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()
    authenticity = "FAKE" if predicted_label == 1 else "REAL"
    return authenticity

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news_text = request.form['news_text']
        
        # Preprocess text
        preprocessed_text = preprocess_text(news_text)
        
        # Predict using traditional ML models
        rfc_prediction = rfc_model.predict([preprocessed_text])[0]
        mlp_prediction = mlp_model.predict([preprocessed_text])[0]
        lr_prediction = lr_model.predict([preprocessed_text])[0]
        dt_prediction = dt_model.predict([preprocessed_text])[0]
        gbc_prediction = gbc_model.predict([preprocessed_text])[0]
        mnb_prediction = mnb_model.predict([preprocessed_text])[0]
        
        # Predict using BERT model
        bert_prediction = predict_with_bert(news_text)
        
        predictions = {
            'rfc_prediction': rfc_prediction,
            'mlp_prediction': mlp_prediction,
            'lr_prediction': lr_prediction,
            'dt_prediction': dt_prediction,
            'gbc_prediction': gbc_prediction,
            'mnb_prediction': mnb_prediction,
            'bert_prediction': bert_prediction
        }
        return render_template('index.html', predictions=predictions, news_text=news_text)
    return render_template('index.html', predictions=None, news_text='')

if __name__ == '__main__':
    app.run(debug=True)
