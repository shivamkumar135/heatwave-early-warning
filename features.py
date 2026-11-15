import numpy as np
import pandas as pd

df = pd.read_csv("clean_weather.csv", parse_dates=["date"])
df = df.sort_values(["city", "date"]).reset_index(drop=True)

def heat_index_c(t_c, rh):
    t = t_c * 9 / 5 + 32
    simple = 0.5 * (t + 61 + (t - 68) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523*t + 10.14333127*rh
            - 0.22475541*t*rh - 0.00683783*t*t
            - 0.05481717*rh*rh + 0.00122874*t*t*rh
            + 0.00085282*t*rh*rh - 0.00000199*t*t*rh*rh)
    hi_f = np.where((simple + t) / 2 >= 80, full, simple)
    return (hi_f - 32) * 5 / 9

df["heat_index"] = heat_index_c(df["temp_max"], df["humidity_min"])

df["alert"] = pd.cut(
    df["heat_index"],
    bins=[-np.inf, 32, 41, 54, np.inf],
    labels=["Green", "Yellow", "Orange", "Red"],
    right=False,
)

# Feature 1: Hidden Danger Days
df["hidden_danger"] = (df["temp_max"] < 40) & (df["heat_index"] >= 41)

# Feature 2: lagatar garam din
df["hot"] = (df["heat_index"] >= 41).astype(int)
def streak(s):
    grp = (s != s.shift()).cumsum()
    return s.groupby(grp).cumsum()
df["hot_streak"] = df.groupby("city")["hot"].transform(streak)

# Feature 3: Hot Night
df["hot_night"] = df["temp_min"] >= 28

df.to_csv("features.csv", index=False)

summary = df.groupby("city").agg(
    orange_red_days=("hot", "sum"),
    hidden_danger_days=("hidden_danger", "sum"),
    longest_streak=("hot_streak", "max"),
    hot_nights=("hot_night", "sum"),
)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(summary)