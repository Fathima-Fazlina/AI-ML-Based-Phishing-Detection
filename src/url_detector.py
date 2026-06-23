import re
import tldextract
import numpy as np
import joblib
from urllib.parse import urlparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SHORTENERS = {'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly'}

def extract_features(url: str) -> dict:
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    domain = ext.domain + '.' + ext.suffix

    return {
        'url_length': len(url),
        'domain_length': len(domain),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_at': url.count('@'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_slashes': url.count('/'),
        'has_https': int(parsed.scheme == 'https'),
        'has_ip': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', ext.domain))),
        'is_shortener': int(domain in SHORTENERS),
        'subdomain_count': len(ext.subdomain.split('.')) if ext.subdomain else 0,
        'path_length': len(parsed.path),
        'query_length': len(parsed.query),
        'has_suspicious_words': int(
            bool(re.search(r'login|verify|update|secure|account|banking|paypal|ebay', url, re.I))
        ),
    }

def train(df):
    features = df['URL'].apply(extract_features)
    X = pd.DataFrame(list(features))
    y = (df['Label'] == 'bad').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, 'models/url_model.pkl')
    return model

def predict(url: str, model=None) -> dict:
    if model is None:
        model = joblib.load('models/url_model.pkl')
    df = pd.DataFrame([extract_features(url)])
    prob = model.predict_proba(df)[0][1]
    return {'url': url, 'phishing_probability': round(prob, 4), 'is_phishing': prob > 0.5}