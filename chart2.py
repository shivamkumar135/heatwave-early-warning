import pandas as pd
import plotly.express as px

df = pd.read_csv("features.csv", parse_dates=["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["hidden_danger"] = df["hidden_danger"].astype(int)
df["hot_night"] = df["hot_night"].astype(int)

# Chart 4: Alert level share (%)
colors = {"Green": "#2ca02c", "Yellow": "#f2c200",
          "Orange": "#ff7f0e", "Red": "#d62728"}
a = df.groupby(["city", "alert"]).size().reset_index(name="days")
a["pct"] = a["days"] / a.groupby("city")["days"].transform("sum") * 100
fig4 = px.bar(
    a, x="city", y="pct", color="alert",
    color_discrete_map=colors,
    category_orders={"alert": ["Green", "Yellow", "Orange", "Red"]},
    title="Alert level share: saal ke kitne % din kis level mein",
    labels={"pct": "% din"},
)
fig4.write_html("chart4_alert_share.html")
fig4.show()

# Chart 5: Hidden Danger Days, city vs mahina
h = df.pivot_table(index="city", columns="month",
                   values="hidden_danger", aggfunc="sum")
fig5 = px.imshow(
    h, color_continuous_scale="YlOrRd", aspect="auto", text_auto=True,
    title="Hidden Danger Days: kis mahine temperature alert sabse zyada miss karta hai",
    labels={"x": "Mahina", "y": "City", "color": "Din"},
)
fig5.update_xaxes(dtick=1)
fig5.write_html("chart5_hidden_by_month.html")
fig5.show()

# Chart 6: Temperature vs Heat Index
sm = df.sample(4000, random_state=1)
fig6 = px.scatter(
    sm, x="temp_max", y="heat_index", color="humidity_min",
    color_continuous_scale="Viridis", opacity=0.6,
    title="Temperature vs Heat Index: humidity safe din ko khatarnak bana deti hai",
    labels={"temp_max": "Max temperature (C)",
            "heat_index": "Heat Index (C)",
            "humidity_min": "Humidity %"},
)
fig6.add_hline(y=41, line_dash="dash", line_color="red")
fig6.add_vline(x=40, line_dash="dash", line_color="black")
fig6.write_html("chart6_scatter.html")
fig6.show()

# Chart 7: Hot Nights per year
n = df.groupby(["year", "city"])["hot_night"].sum().reset_index()
fig7 = px.line(
    n, x="year", y="hot_night", color="city", markers=True,
    title="Hot Nights (min temp >= 28C) 2015-2025",
    labels={"hot_night": "Raatein"},
)
fig7.write_html("chart7_hot_nights.html")
fig7.show()