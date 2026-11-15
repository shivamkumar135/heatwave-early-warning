import pandas as pd

df = pd.read_csv("raw_weather.csv")

df = df.rename(columns={
    "time": "date",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "relative_humidity_2m_mean": "humidity",
    "relative_humidity_2m_min": "humidity_min",
})

df["date"] = pd.to_datetime(df["date"])

print("Missing values:")
print(df.isna().sum())

df = df.dropna()
df = df.drop_duplicates()

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df.to_csv("clean_weather.csv", index=False)

print("Final shape:", df.shape)
print(df.describe())
print("Sabse zyada temperature:")
print(df.groupby("city")["temp_max"].max())