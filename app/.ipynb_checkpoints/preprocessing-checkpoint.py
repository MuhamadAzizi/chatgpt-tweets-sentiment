import re
import html
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessing:
    def text_filtering(self, text):
        text = text.lower()
        text = html.unescape(text)  # Remove HTML escape sequences
        text = re.sub(r'[\n\t\r\\]', ' ', text)  # Remove escape sequences
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text) # Remove emails
        text = re.sub('['
                      u'\U0001F600-\U0001F64F'  # Emoticons
                      u'\U0001F300-\U0001F5FF'  # Symbols & pictographs
                      u'\U0001F680-\U0001F6FF'  # Transport & map symbols
                      u'\U0001F1E0-\U0001F1FF'  # Flags (iOS)
                      ']+', '', text)  # Remove emojis
        text = re.sub(r'@\w+', '', text)  # Remove usernames
        text = re.sub(r'#[\w\d]+', '', text)  # Remove hashtags
        text = re.sub(r'\'s', ' is', text)  # Replace possessive forms

        # Expand contractions
        contraction_patterns = [
            (r"\bcan't\b", "cannot"),
            (r"\bwon't\b", "will not"),
            (r"\bain't\b", "am not"),
            (r"\bisn't\b", "is not"),
            (r"\baren't\b", "are not"),
            (r"\bwasn't\b", "was not"),
            (r"\bweren't\b", "were not"),
            (r"\bhasn't\b", "has not"),
            (r"\bhaven't\b", "have not"),
            (r"\bhadn't\b", "had not"),
            (r"\bdoesn't\b", "does not"),
            (r"\bdon't\b", "do not"),
            (r"\bdidn't\b", "did not"),
            (r"\bcouldn't\b", "could not"),
            (r"\bshouldn't\b", "should not"),
            (r"\bwouldn't\b", "would not"),
            (r"\bmightn't\b", "might not"),
            (r"\bmustn't\b", "must not"),
            (r"\bshan't\b", "shall not"),
            (r"\bi'm\b", "I am"),
            (r"\byou're\b", "you are"),
            (r"\bhe's\b", "he is"),
            (r"\bshe's\b", "she is"),
        ]

        for pattern, replacement in contraction_patterns:
            text = re.sub(pattern, replacement, text)

        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                      r'|www(?:\.[a-zA-Z0-9-]+){2,3}(?:/[a-zA-Z0-9]+)?', '', text)  # Remove links
        text = re.sub(r'[/-]', ' ', text)  # Replace slashes and dashes with spaces
        text = re.sub(r'\$\w*\b', '', text)  # Remove dollar words
        text = re.sub(r'[^\w\s]+', '', text)  # Remove punctuation
        text = re.sub(r'\b\w*\d+\w*\b', '', text)  # Remove words containing digits
        text = re.sub(r'\b(?!i\b)\w\b', ' ', text)  # Remove one-character words (except 'I')
        text = re.sub(r'[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF'
                      r'\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF\u0E00-\u0E7F\u0E80-\u0EFF'
                      r'\u0F00-\u0FFF]+', '', text)  # Remove Indian script characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = re.sub(r'[\u2000-\u206F\u2E00-\u2E7F\\!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]', '', text)  # Remove special Unicode characters
        text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip()  # Remove leading and trailing spaces

        return text
    
    def tokenization(self, text):
        return word_tokenize(text)
        
    def remove_stopwords(self, token):
        return [word for word in token if word not in stopwords.words('english')]
    
    def stemming(self, token):
        ps = PorterStemmer()
        return [ps.stem(word) for word in token]
    
    def vector_conversion(self, token):
        vectorizer = joblib.load('app/vectorizer.joblib')
        vector = vectorizer.transform([' '.join(token)])
        return np.asarray(vector.toarray()[0]).reshape(1, -1)
    
    def label_decoder(self, y):
        if y[0] == 0:
            return 'Negative'
        elif y[0] == 1:
            return 'Neutral'
        elif y[0] == 2:
            return 'Positive'