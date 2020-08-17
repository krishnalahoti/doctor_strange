
import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

a = "This is a good movie"
sid.polarity_scores(a)

a = "This was the best, most awesome movie, I LOVED IT!!!"
sid.polarity_scores(a)

import pandas as pd

df = pd.read_csv('../TextFiles/amazonreviews.tsv',sep = '\t')

df.head()

df.isnull().sum()

# code to check if there is a blank reviee.

blanks = []

for i,lb,rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)


blanks
sid.polarity_scores(df.iloc[0]['review'])

df['scores'] = df['review'].apply(lambda review:sid.polarity_scores(review))

df['compound'] = df['scores'].apply(lambda d:d['compound'])

df['compound_score'] = df['compound'].apply(lambda score:'pos' if score >=0 else 'neg')

df.head()

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


accuracy_score(df['label'],df['compound_score'])

confusion_matrix(df['label'],df['compound_score'])

print(classification_report(df['label'],df['compound_score']))
