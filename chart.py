import pandas as pd
import plotly.express as px

df = pd.read_csv("features.csv", parse_dates=["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Chart 1: Temperature alert ne kitne khatarnak din pakde aur kitne miss kiye
s = df.groupby("city").agg(
    total=("hot", "sum"), hidden=("hidden_danger", "sum")
).reset_index()
s["caught"] = s["total"] - s["hidden"]
fig1 = px.bar(
    s, x="city", y=["caught", "hidden"], barmode="stack",
    title="Khatarnak din: temperature alert ne pakde vs miss kiye",
    labels={"value": "Din", "variable": "Type"},
)
fig1.write_html("chart1_hidden.html")
fig1.show()

# Chart 2: Saal-dar-saal khatarnak din
t = df.groupby(["year", "city"])["hot"].sum().reset_index()
fig2 = px.line(
    t, x="year", y="hot", color="city", markers=True,
    title="Khatarnak din (Heat Index >= 41) 2015-2025",
    labels={"hot": "Din"},
)
fig2.write_html("chart2_trend.html")
fig2.show()

# Chart 3: Har city ka heatmap
for city in df["city"].unique():
    p = df[df["city"] == city].pivot_table(
        index="year", columns="month", values="heat_index", aggfunc="mean"
    )
    fig = px.imshow(
        p, color_continuous_scale="YlOrRd", aspect="auto",
        zmin=15, zmax=50,
        title=f"{city}: Average Heat Index (saal vs mahina)",
        labels={"x": "Mahina", "y": "Saal", "color": "Heat Index"},
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
    fig.write_html(f"heatmap_{city}.html")
    fig.show()
