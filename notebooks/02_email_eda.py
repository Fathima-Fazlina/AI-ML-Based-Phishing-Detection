import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re


def load_emails_from_folder(folder_path, label):
    emails = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
                emails.append({'text': content, 'label': label})
        except Exception:
            pass
    return emails

spam_path = r'C:\Users\Fathima Fazlina\phishing-detector\data\raw\emails\spam'
ham_path = r'C:\Users\Fathima Fazlina\phishing-detector\data\raw\emails\easy_ham'
#os.listdir(folder_path)

spam_emails = load_emails_from_folder(spam_path, label=1)
ham_emails  = load_emails_from_folder(ham_path,  label=0)

df = pd.DataFrame(spam_emails + ham_emails)
print("Total emails:", len(df))
print("Spam:", df['label'].sum())
print("Ham: ", (df['label'] == 0).sum())
df.head(2)

df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, name, color in [(0, 'Ham', 'steelblue'), (1, 'Spam', 'crimson')]:
    subset = df[df['label'] == label]
    axes[0].hist(subset['char_count'].clip(upper=10000), bins=50,
                 alpha=0.6, label=name, color=color)
    axes[1].hist(subset['word_count'].clip(upper=2000), bins=50,
                 alpha=0.6, label=name, color=color)

for ax, title in zip(axes, ['Character Count', 'Word Count']):
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

spam_text = df[df['label'] == 1]['text'].tolist()
ham_text  = df[df['label'] == 0]['text'].tolist()

vectorizer = CountVectorizer(stop_words='english', max_features=30,
                              ngram_range=(1,1), min_df=5)
vectorizer.fit(spam_text)
spam_counts = vectorizer.transform(spam_text).toarray().sum(axis=0)
spam_words  = pd.Series(spam_counts, index=vectorizer.get_feature_names_out())

plt.figure(figsize=(12, 5))
spam_words.sort_values(ascending=True).tail(20).plot(kind='barh', color='crimson', alpha=0.8)
plt.title('Top 20 Words in Spam Emails')
plt.xlabel('Total occurrences')
plt.tight_layout()
plt.show()

# Keep only text and label, shuffle
df_clean = df[['text', 'label']].sample(frac=1, random_state=42).reset_index(drop=True)
df_clean.to_csv('../data/processed/emails.csv', index=False)
print("Saved to data/processed/emails.csv")
print("Shape:", df_clean.shape)



df = pd.DataFrame(spam_emails + ham_emails)

print("Total emails:", len(df))

from src.email_detector import train

model = train(df['text'], df['label'])
print("Email model saved successfully!")