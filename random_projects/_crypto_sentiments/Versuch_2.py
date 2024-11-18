import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid", palette="viridis")

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)

"""#   data-preprocessing  """
crypto_df = pd.read_csv("cryptonews.csv")
bitcoin_df = pd.read_csv("Bitcoin History.csv")

#   format date
crypto_df["date"] = pd.to_datetime(crypto_df["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
#print(crypto_df["date"].isnull().sum())
crypto_df.dropna(subset=["date"], inplace=True)
crypto_df["date"] = crypto_df["date"].dt.strftime("%Y-%m-%d")
#print(crypto_df["date"])
#print(crypto_df.info())


"""#   Überblick über alle Werte!"""
def summary(df):
    summary_df = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary_df['missing#'] = df.isna().sum()
    summary_df['missing%'] = df.isna().sum() / len(df)
    summary_df['count'] = df.count().values
    summary_df['unique'] = df.nunique().values
    return summary_df

#   nice display!! (on juypter)
summary(crypto_df).style.background_gradient(cmap='Purples')


#   format date
bitcoin_df.rename(columns={"Date": "date", "Change %": "change"}, inplace=True)
bitcoin_df["date"] = pd.to_datetime(bitcoin_df["date"]).dt.strftime("%Y-%m-%d")
#print(bitcoin_df.describe().T)

#   select only the dates that are relevant from other dataframe
#print(crypto_df["date"].min())
#print(crypto_df["date"].max())
bitcoin_df = bitcoin_df[(bitcoin_df["date"] >= "2021-10-12") & (bitcoin_df["date"] <= "2023-12-19")]
#print(crypto_df.describe().T)
#print(bitcoin_df.describe().T)
#print(bitcoin_df)

#   nice display!! (on juypter)
summary(bitcoin_df).style.background_gradient(cmap="Oranges")

#   relevant info in bitcoin_df
#print(bitcoin_df)
def clean_change(change):
    return float(str(change)[:-1])

bitcoin_df = bitcoin_df[["date", "change"]]
bitcoin_df["change"] = bitcoin_df["change"].apply(clean_change)
#print(bitcoin_df.head())


#   extract sentiment out of string in crypto_df
#print(crypto_df.loc[0]["sentiment"])
def get_sentiment(data):
    sentiment = eval(data)
    return sentiment["class"]
crypto_df["sentiment"] = crypto_df["sentiment"].apply(get_sentiment)
#print(crypto_df.head())

#   printo out all possible values
print("Possible values: ")
print(f"Sentiment: {crypto_df.sentiment.unique()}")
print(f"Source: {crypto_df.source.unique()}")
print(f"Subject: {crypto_df.subject.unique()}")

#   Add three new columns to count sentiments AND drop rest useless columns
crypto_sentiments = crypto_df[["date", "sentiment", "source", "subject"]]

crypto_sentiments["negative"] = (crypto_sentiments["sentiment"] == "negative").astype(int)
crypto_sentiments["neutral"] = (crypto_sentiments["sentiment"] == "neutral").astype(int)
crypto_sentiments["positive"] = (crypto_sentiments["sentiment"] == "positive").astype(int)
crypto_sentiments.drop(columns=["sentiment", "source", "subject"], inplace = True)
#print(crypto_sentiments.head())
#print(bitcoin_df.head())

#   nice summary!
crypto_sentiments_to_dates = crypto_sentiments.groupby("date").sum().reset_index().sort_values(by="date")
#print(crypto_sentiments_to_dates.head(10))

""" #   merge data   """
#print(crypto_sentiments_to_dates.info())
#print(bitcoin_df.info())
df = pd.merge(crypto_sentiments_to_dates, bitcoin_df, on="date", how="inner")
print(df.head(10))


"""     ML Model Regression on neg / neutral / pos sentiments. Labels = change"""

"""         TTENSORFLOW Neural network      """

"""     Das habe ich noch nicht studiert!! Noch ein TODO!!      """
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = df.drop(['date', 'change'], axis=1)
target = df['change']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer='adam', loss='mean_squared_error')


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)



model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])


mse = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on Test Set: {mse}')


predictions = model.predict(X_test_scaled)


results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
print(results_df.head(10))

"""     Visualize the Results       """
plt.figure(figsize=(12, 8), dpi=300)
sns.scatterplot(data=results_df, x='Actual', y='Predicted')
plt.xlabel('Actual Change')
plt.ylabel('Predicted Change')
plt.title('Actual vs Predicted Change')
plt.tight_layout()
plt.show()

"""     Save Model      """
#model.save('models/crypto_nb.h5')