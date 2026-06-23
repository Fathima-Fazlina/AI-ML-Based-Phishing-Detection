import email
import re
import joblib
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

PHISHING_PATTERNS = [
    r'verify your account', r'click here to confirm', r'your account (has been|will be) (suspended|limited)',
    r'update your (payment|billing|credit card)', r'won a prize', r'act (now|immediately)',
    r'dear (customer|user|member)', r'unusual (sign.in|activity|login)',
]

def extract_email_text(raw_email: str) -> str:
    try:
        msg = email.message_from_string(raw_email)
    except Exception:
        return raw_email

    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            elif part.get_content_type() == 'text/html':
                soup = BeautifulSoup(part.get_payload(decode=True), 'html.parser')
                body += soup.get_text()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode('utf-8', errors='ignore')
    return body.strip()

def extract_features(text: str) -> dict:
    lower = text.lower()
    return {
        'pattern_hits': sum(1 for p in PHISHING_PATTERNS if re.search(p, lower)),
        'exclamation_count': text.count('!'),
        'url_count': len(re.findall(r'https?://', text)),
        'word_count': len(text.split()),
        'has_urgent': int(bool(re.search(r'urgent|immediately|asap|within 24', lower))),
    }

def train(texts, labels):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, C=1.0)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    print(classification_report(y_test, pipeline.predict(X_test)))
    joblib.dump(pipeline, 'models/email_model.pkl')
    return pipeline

def predict(raw_email: str, model=None) -> dict:
    if model is None:
        model = joblib.load('models/email_model.pkl')
    df = extract_email_text(raw_email)
    prob = model.predict_proba([df])[0][1]
    return {'phishing_probability': round(prob, 4), 'is_phishing': prob > 0.5, 'body_preview': df[:200]}
