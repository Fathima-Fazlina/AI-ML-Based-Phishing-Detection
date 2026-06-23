import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import tldextract
import re

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.url_detector import train

df = pd.read_csv(r'data/raw/phishing_site_urls.csv')
model = train(df)

df = pd.read_csv(r'C:\Users\Fathima Fazlina\phishing-detector\data\raw\phishing_site_urls.csv')
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nLabel counts:\n", df['Label'].value_counts())
print("\nSample rows:")
df.head()

print("Missing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Standardize label column
df['label'] = df['Label'].map({'good': 0, 'bad': 1})
print("\nAfter mapping:", df['label'].value_counts())

df['url_length'] = df['URL'].apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution by class
for label, name, color in [(0, 'Legitimate', 'steelblue'), (1, 'Phishing', 'crimson')]:
    subset = df[df['label'] == label]['url_length']
    axes[0].hist(subset, bins=60, alpha=0.6, label=name, color=color)
axes[0].set_title('URL Length Distribution')
axes[0].set_xlabel('URL Length')
axes[0].set_ylabel('Count')
axes[0].legend()

# Box plot
df.boxplot(column='url_length', by='Label', ax=axes[1])
axes[1].set_title('URL Length by Class')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Length')

plt.tight_layout()
plt.show()

print("Avg URL length — Phishing:", df[df['label']==1]['url_length'].mean().round(1))
print("Avg URL length — Legit:   ", df[df['label']==0]['url_length'].mean().round(1))

def quick_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    return {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_digits': sum(c.isdigit() for c in url),
        'has_https': int(parsed.scheme == 'https'),
        'has_ip': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', ext.domain or ''))),
        'subdomain_count': len(ext.subdomain.split('.')) if ext.subdomain else 0,
        'path_length': len(parsed.path),
        'has_suspicious_word': int(bool(re.search(
            r'login|verify|update|secure|account|banking|paypal|ebay', url, re.I
        ))),
    }

print("Extracting features... (takes ~1 min for large datasets)")
features_df = df['URL'].apply(quick_features).apply(pd.Series)
features_df['label'] = df['label'].values
print("Done. Shape:", features_df.shape)
features_df.head()

plt.figure(figsize=(10, 7))
corr = features_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5)
plt.title('Feature Correlation with Phishing Label')
plt.tight_layout()
plt.show()

phishing_urls = df[df['label'] == 1]['URL']

print("=== Common patterns in phishing URLs ===\n")

# IP addresses
ip_urls = phishing_urls[phishing_urls.str.contains(r'\d+\.\d+\.\d+\.\d+', regex=True)]
print(f"URLs with IP address: {len(ip_urls)} ({len(ip_urls)/len(phishing_urls)*100:.1f}%)")

# Suspicious keywords
for word in ['login', 'verify', 'account', 'secure', 'update', 'paypal', 'banking']:
    count = phishing_urls.str.lower().str.contains(word).sum()
    print(f"Contains '{word}': {count} ({count/len(phishing_urls)*100:.1f}%)")

# HTTP vs HTTPS
http_count = phishing_urls.str.startswith('http://').sum()
print(f"\nUsing HTTP (not HTTPS): {http_count} ({http_count/len(phishing_urls)*100:.1f}%)")

features_df.to_csv('../data/processed/url_features.csv', index=False)
print("Saved to data/processed/url_features.csv")
print("Shape:", features_df.shape)
print("\nClass balance:")
print(features_df['label'].value_counts(normalize=True).round(3))