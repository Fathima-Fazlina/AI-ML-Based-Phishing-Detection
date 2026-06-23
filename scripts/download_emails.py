import urllib.request, tarfile, os, shutil

os.makedirs('data/raw/emails/spam', exist_ok=True)
os.makedirs('data/raw/emails/ham', exist_ok=True)

urls = [
    ('https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2', 'spam'),
    ('https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2', 'ham'),
]

for url, label in urls:
    fname = f'data/raw/{label}.tar.bz2'
    print(f'Downloading {label}...')
    urllib.request.urlretrieve(url, fname)
    print(f'Extracting {label}...')
    with tarfile.open(fname, 'r:bz2') as tar:
        tar.extractall(f'data/raw/emails/{label}_extracted')
    print(f'Done: {label}')