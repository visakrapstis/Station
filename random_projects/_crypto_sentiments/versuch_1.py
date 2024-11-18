import pandas as pd
import json

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
"""pd.options.display.max_seq_items = None"""


data = pd.read_csv("cryptonews.csv")

"""#Allgemein"""
#print(data.shape)
#print(data.dtypes)

"""detenbereinigung."""
#print(data.columns)

#   1. unnötiger Spalte
df2 = data.drop(["url"], axis=1)

#   2.  sentiment-spalte ist die wichtigste, mit labels, polarities, subjectivities. Die Spalte ist aber ein String
#Konvertieren der Spalte ins json, Herstellen 3 neuer Spalten entsprechend. (Könnte man mit apply)
#   !!! ggf gibt es funktion in pandas DF to_json!
df3 = df2.copy()
classes = []
polarities = []
subjectivities = []

#   Diese teil kann man mit list-comprehension schnell bereinigen!
for i in df2.sentiment:
    sentiment = json.loads(str(i).replace("'", '"'))["class"]
    classes.append(sentiment)
    polarity = json.loads(str(i).replace("'", '"'))["polarity"]
    polarities.append(polarity)
    subjectivity = json.loads(str(i).replace("'", '"'))["subjectivity"]
    subjectivities.append(subjectivity)

df3["classes"] = classes
df3["polarity"] = polarities
df3["subjectivity"] = subjectivities

#   3.  drop String and convert classes into 'sentiments'
df4 = df3.drop(["sentiment"], axis=1)
df4["sentiment"] = df4["classes"]
df4.drop(["classes"], axis=1, inplace=True)
#print(df4[["polarity", "subjectivity", "sentiment"]])
#print(df4.dtypes)

##  4.  checking for null values
#print(df4.isnull().sum())

#   5.  convert sentiments to 0,1,2 (neg, neutral, pos) with apply
def moods(x):
    if str(x) == "neutral":
        return 1
    elif str(x) == "negative":
        return 0
    else:
        return 2
df5 = df4.copy()
df5["sentiment"] = df4["sentiment"].apply(moods)
print(df5.columns)
#(df5["sentiment"].unique())

#print(df5[["subject", "title", "polarity", "subjectivity", "sentiment"]].head())
#print(df5.title[:20])

#daten - zwischen - visualisierung
"""
sentiments from sources,
sentiments on coins
sentiments according to time!! very important (leider haben wir nur 2 Jahre DAten)..
"""
#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.set_theme(style="whitegrid")

#sns.pairplot(data=df5, hue="sentiment")

#   Mean polarity for a coin per month!
#months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
#df6 = df5.copy()
#df6["subject"] = df6["subject"].str.contains("bitcoin")
#for i in months:
#    i = f"2022-{i}"
#    print(df6[df6["date"]>i].polarity.mean())

#   Automatic report on data. Here did not work, maybe because of insufficient data..
#import ydata_profiling as pp
#pp.ProfileReport(df5)


#s. Jupyter.. keine guuter Visualisierung möglich bei sentiment Daten...!!

"""weitere datenbearbeitung für ML"""
#drop date, source, subject and title +!! [polarity and subjectivity] as the data is not relevant. Separate labels in new frame
df6 = df5["text"]
y = df5["sentiment"]


"""ML-Model"""
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df6)
#print(X[:20])


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

NB = MultinomialNB()
NB.fit(X_train, y_train)
predictions = NB.predict(X_test)
#print(predictions)
#print(y_test)


"""testing"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


def predict_sentiment(string, model):
    prediction = model.predict(vectorizer.transform(string))
    if prediction == 0:
        return "negative"
    elif prediction == 1:
        return "neutral"
    else:
        return "positive"

print(predict_sentiment(["never never never buy!!"], NB))
print(NB.predict_proba(vectorizer.transform(["never never never buy!!"])))