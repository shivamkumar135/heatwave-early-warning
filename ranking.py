import pandas as pd
import plotly.express as px

df = pd.read_csv("features.csv", parse_dates=["date"])
years = df["date"].dt.year.nunique()

# Feature 6: City Risk Ranking
r = (df.groupby("city")["hot"].sum() / years).round(1)
r = r.reset_index(name="dangerous_days_per_year")
print(r.sort_values("dangerous_days_per_year", ascending=False))

r = r.sort_values("dangerous_days_per_year")
fig = px.bar(
    r, x="dangerous_days_per_year", y="city", orientation="h",
    title="City Risk Ranking: saal mein ausat khatarnak din",
    labels={"dangerous_days_per_year": "Din per saal"},
)
fig.write_html("chart8_ranking.html")
fig.show()

# Feature 4: Group-wise Advice
advice = {
    "Green":  {"Outdoor worker": "Normal kaam", "Bujurg": "Normal", "Bachche": "Normal khel-kood"},
    "Yellow": {"Outdoor worker": "Har ghante paani, chhaon mein break", "Bujurg": "Dhoop se bachein, paani peete rahein", "Bachche": "Dopahar mein bahar khelna kam karein"},
    "Orange": {"Outdoor worker": "12 se 4 baje kaam band ya kam", "Bujurg": "Ghar ke andar rahein", "Bachche": "Ghar ke andar khelein"},
    "Red":    {"Outdoor worker": "Kaam rokein, chhaya aur paani zaroori", "Bujurg": "Ghar se na niklein, doctor ka number paas rakhein", "Bachche": "Bahar bilkul nahi"},
}
print(pd.DataFrame(advice).T.to_string())