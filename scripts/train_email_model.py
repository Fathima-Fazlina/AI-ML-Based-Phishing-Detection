import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load emails from folder ──────────────────────────────────────────────────
def load_folder(path, label):
    emails = []
    if not os.path.exists(path):
        print(f"WARNING: folder not found → {path}")
        return emails
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='latin-1') as f:
                emails.append({'text': f.read(), 'label': label})
        except Exception:
            pass
    print(f"Loaded {len(emails)} emails from {path}")
    return emails

# ── Try multiple folder layouts ───────────────────────────────────────────────
spam_candidates = [
    'data/raw/emails/spam',
    'data/raw/emails/spam_extracted/spam',
    'data/raw/emails/spam_extracted',
]
ham_candidates = [
    'data/raw/emails/ham',
    'data/raw/emails/ham_extracted/easy_ham',
    'data/raw/emails/ham_extracted',
]

spam, ham = [], []
for p in spam_candidates:
    spam = load_folder(p, 1)
    if spam:
        break

for p in ham_candidates:
    ham = load_folder(p, 0)
    if ham:
        break

if not spam and not ham:
    # ── Fallback: build a tiny synthetic dataset so the app still works ───────
    print("\nNo email files found — building synthetic training data as fallback...")
    spam = [
        {'text': 'URGENT: Verify your account immediately or it will be suspended. Click here to confirm your details now!', 'label': 1},
        {'text': 'Congratulations! You have won a $1000 prize. Act now and claim your reward by clicking the link below.', 'label': 1},
        {'text': 'Dear Customer, your PayPal account has been limited. Please update your billing information immediately.', 'label': 1},
        {'text': 'WARNING: Unusual sign-in activity detected on your account. Verify your identity now to avoid suspension.', 'label': 1},
        {'text': 'Your bank account requires verification. Login now at http://192.168.1.1/secure to confirm your details.', 'label': 1},
        {'text': 'You have been selected for a special offer! Click here to get your free gift card now. Limited time only!', 'label': 1},
        {'text': 'Dear user, your account will be deactivated unless you verify your email address within 24 hours.', 'label': 1},
        {'text': 'ALERT: Your credit card payment failed. Update your payment details immediately to avoid service interruption.', 'label': 1},
        {'text': 'Win a brand new iPhone! You are our lucky winner today. Click below to claim your prize instantly.', 'label': 1},
        {'text': 'Your Netflix account has been suspended. Verify your billing info now at our secure portal to restore access.', 'label': 1},
    ] * 20  # repeat to get 200 samples

    ham = [
        {'text': 'Hi, just checking in on the project status. Can we schedule a meeting for next week?', 'label': 0},
        {'text': 'The quarterly report has been uploaded to the shared drive. Please review before Friday.', 'label': 0},
        {'text': 'Thanks for your order! Your package has been shipped and will arrive in 3-5 business days.', 'label': 0},
        {'text': 'Reminder: Team standup at 10am tomorrow. Please update your task board before the meeting.', 'label': 0},
        {'text': 'Your subscription has been renewed. Your next billing date is July 1st. Thank you for staying with us.', 'label': 0},
        {'text': 'Hey, are you free for lunch on Thursday? Let me know and I can book a table.', 'label': 0},
        {'text': 'The new feature release is scheduled for next Monday. All tests are passing and we are good to go.', 'label': 0},
        {'text': 'Please find attached the invoice for last month services. Let me know if you have any questions.', 'label': 0},
        {'text': 'Your appointment is confirmed for June 25th at 2pm. Please bring a valid ID to the clinic.', 'label': 0},
        {'text': 'Great news! The proposal has been approved by the board. We can proceed with the next phase.', 'label': 0},
    ] * 20  # repeat to get 200 samples

    print(f"Synthetic dataset: {len(spam)} spam + {len(ham)} ham")

# ── Build dataframe ──────────────────────────────────────────────────────────
df = pd.DataFrame(spam + ham).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nTotal samples: {len(df)}")
print(f"Spam: {df['label'].sum()}  |  Ham: {(df['label']==0).sum()}")

# ── Train ─────────────────────────────────────────────────────────────────────
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1
    )),
    ('clf', LogisticRegression(max_iter=1000, C=1.0)),
])

print("\nTraining...")
pipeline.fit(X_train, y_train)

print("\nEvaluation on test set:")
print(classification_report(y_test, pipeline.predict(X_test)))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/email_model.pkl')
print("\nModel saved → models/email_model.pkl")