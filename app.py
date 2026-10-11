import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Heatwave Early Warning", layout="wide")

COORDS = {
    "Patna": (25.59, 85.14), "Gaya": (24.80, 85.00),
    "Lucknow": (26.85, 80.95), "Jaipur": (26.91, 75.79),
    "Guwahati": (26.14, 91.74), "Kolkata": (22.57, 88.36),
    "Bengaluru": (12.97, 77.59),
}
COLORS = {"Green": "#2ca02c", "Yellow": "#f2c200",
          "Orange": "#ff7f0e", "Red": "#8b0000"}
ORDER = ["Green", "Yellow", "Orange", "Red"]
LEGEND = [
    ("Green", "Heat Index below 32", "Normal, no risk"),
    ("Yellow", "32 to 41", "Be careful"),
    ("Orange", "41 to 54", "Dangerous, avoid going out"),
    ("Red", "Above 54", "Extremely dangerous"),
]
# (outdoor worker, elderly, children)
ADVICE = {
    "Green": ("Normal work", "Normal", "Normal play"),
    "Yellow": ("Drink water every hour, take breaks in the shade",
               "Avoid direct sun, keep drinking water",
               "Reduce outdoor play at midday"),
    "Orange": ("Stop or reduce work between 12 pm and 4 pm",
               "Stay indoors", "Play indoors"),
    "Red": ("Stop work, make sure shade and water are available",
            "Do not go out, keep a doctor's number handy",
            "No outdoor activity"),
}


def legend_html(direction="row"):
    html = (f"<div style='display:flex;flex-direction:{direction};"
            "gap:8px;flex-wrap:wrap;margin:6px 0 12px 0'>")
    for name, rng, txt in LEGEND:
        html += (
            "<div style='display:flex;align-items:center;gap:8px;"
            "padding:6px 12px;border-radius:8px;"
            "background:rgba(128,128,128,0.15)'>"
            f"<span style='width:16px;height:16px;border-radius:50%;"
            f"background:{COLORS[name]};display:inline-block'></span>"
            f"<span><b>{name}</b>: {rng} ({txt})</span></div>")
    return html + "</div>"


@st.cache_data
def load():
    df = pd.read_csv("features.csv", parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def heat_index_c(t_c, rh):
    t = t_c * 9 / 5 + 32
    simple = 0.5 * (t + 61 + (t - 68) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523*t + 10.14333127*rh
            - 0.22475541*t*rh - 0.00683783*t*t
            - 0.05481717*rh*rh + 0.00122874*t*t*rh
            + 0.00085282*t*rh*rh - 0.00000199*t*t*rh*rh)
    hi_f = np.where((simple + t) / 2 >= 80, full, simple)
    return (hi_f - 32) * 5 / 9


def level(hi):
    if hi < 32:
        return "Green"
    if hi < 41:
        return "Yellow"
    if hi < 54:
        return "Orange"
    return "Red"


@st.cache_data(ttl=3600)
def get_forecast(lat, lon, days):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,relative_humidity_2m_min",
        "forecast_days": days, "timezone": "Asia/Kolkata"}, timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json()["daily"])


df = load()
cities = sorted(df["city"].unique())
city = st.sidebar.selectbox("Select city", cities,
                            index=cities.index("Kolkata"))
st.sidebar.markdown("**What the alert colors mean**")
st.sidebar.markdown(legend_html("column"), unsafe_allow_html=True)
d = df[df["city"] == city]

st.title("Heatwave Early Warning & Heat Stress Index")
st.caption("Data: Open-Meteo (ERA5 reanalysis), 2015-2025")

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Overview", "Hidden Danger", "Trend", "Ranking",
     "Forecast & Advice", "Map"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dangerous days", int(d["hot"].sum()))
    c2.metric("Hidden Danger days", int(d["hidden_danger"].sum()))
    c3.metric("Longest hot streak (days)", int(d["hot_streak"].max()))
    c4.metric("Hot nights", int(d["hot_night"].sum()))
    st.caption("Dangerous day = Heat Index of 41 or more. Hot night = "
               "minimum temperature of 28 C or more.")
    share = d["alert"].value_counts().reset_index()
    share.columns = ["alert", "days"]
    st.plotly_chart(px.pie(share, names="alert", values="days",
                           color="alert", color_discrete_map=COLORS,
                           title=f"{city}: share of days by alert level"))
    st.markdown(legend_html(), unsafe_allow_html=True)

with t2:
    st.write("**Hidden Danger** = max temperature below 40 C, but Heat Index "
             "41 or more. A temperature-only alert misses these days.")
    sm = d.sample(min(3000, len(d)), random_state=1)
    fig = px.scatter(
        sm, x="temp_max", y="heat_index", color="humidity_min",
        color_continuous_scale="Viridis", opacity=0.6,
        title=f"{city}: max temperature vs Heat Index",
        labels={"temp_max": "Max temperature (C)",
                "heat_index": "Heat Index (C)",
                "humidity_min": "Min humidity (%)"})
    fig.add_hline(y=41, line_dash="dash", line_color="red")
    fig.add_vline(x=40, line_dash="dash", line_color="black")
    st.plotly_chart(fig)
    st.caption("Red line: Heat Index 41. Black line: temperature 40 C. "
               "Points above the red line and left of the black line are "
               "Hidden Danger days.")
    hm = d.groupby("month")["hidden_danger"].sum().reset_index()
    st.plotly_chart(px.bar(
        hm, x="month", y="hidden_danger",
        title=f"{city}: Hidden Danger days by month",
        labels={"month": "Month", "hidden_danger": "Hidden Danger days"}))

with t3:
    yr = d.groupby("year")["hot"].sum().reset_index()
    st.plotly_chart(px.line(
        yr, x="year", y="hot", markers=True,
        title=f"{city}: dangerous days per year",
        labels={"year": "Year", "hot": "Dangerous days"}))
    p = d.pivot_table(index="year", columns="month",
                      values="heat_index", aggfunc="mean")
    fh = px.imshow(p, color_continuous_scale="YlOrRd", aspect="auto",
                   zmin=15, zmax=50,
                   title=f"{city}: average Heat Index by year and month",
                   labels={"x": "Month", "y": "Year",
                           "color": "Heat Index (C)"})
    fh.update_xaxes(dtick=1)
    fh.update_yaxes(dtick=1)
    st.plotly_chart(fh)

with t4:
    r = (df.groupby("city")["hot"].sum() / df["year"].nunique()).round(1)
    r = r.reset_index(name="days_per_year").sort_values("days_per_year")
    st.plotly_chart(px.bar(
        r, x="days_per_year", y="city", orientation="h",
        title="City risk ranking: average dangerous days per year",
        labels={"days_per_year": "Days per year", "city": "City"}))
    s = df.groupby("city").agg(dangerous=("hot", "sum"),
                               missed=("hidden_danger", "sum")).reset_index()
    s["caught"] = s["dangerous"] - s["missed"]
    s = s.rename(columns={"caught": "Caught by temperature alert",
                          "missed": "Missed by temperature alert"})
    st.plotly_chart(px.bar(
        s, x="city",
        y=["Caught by temperature alert", "Missed by temperature alert"],
        title="Dangerous days: caught vs missed by a temperature-only alert",
        labels={"value": "Days", "variable": "", "city": "City"}))

with t5:
    lat, lon = COORDS[city]
    try:
        f = get_forecast(lat, lon, 7)
        f["heat_index"] = heat_index_c(f["temperature_2m_max"],
                                       f["relative_humidity_2m_min"]).round(1)
        f["alert"] = f["heat_index"].apply(level)
        f["temperature-only alert (40 C)"] = np.where(
            f["temperature_2m_max"] >= 40, "Alert", "Safe")
        st.subheader(f"{city}: next 7 days")
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.dataframe(f.rename(columns={
            "time": "date", "temperature_2m_max": "max temp (C)",
            "relative_humidity_2m_min": "min humidity (%)",
            "heat_index": "heat index (C)"}))
        worst = max(f["alert"], key=ORDER.index)
        st.subheader(f"Highest alert this week: {worst}")
        w, e, k = ADVICE[worst]
        st.write(f"**Outdoor workers:** {w}")
        st.write(f"**Elderly people:** {e}")
        st.write(f"**Children:** {k}")
        st.caption("This is sample advice, not an official health guideline.")
    except Exception as ex:
        st.error(f"Could not load the forecast: {ex}")

with t6:
    st.subheader("All cities at once: how risky is each city on a given day?")
    mode = st.radio("What do you want to see?",
                    ["Past date", "Next 7 days (forecast)"], horizontal=True)
    rows = []
    data = {}

    if mode == "Past date":
        day = st.date_input("Select a date", value=pd.Timestamp("2024-06-15"),
                            min_value=pd.Timestamp("2015-01-01"),
                            max_value=pd.Timestamp("2025-12-31"))
        x = df[df["date"] == pd.Timestamp(day)]
        for _, r in x.iterrows():
            la, lo = COORDS[r["city"]]
            rows.append({"city": r["city"], "lat": la, "lon": lo,
                         "max temp": r["temp_max"],
                         "min humidity": r["humidity_min"],
                         "heat index": round(r["heat_index"], 1),
                         "alert": str(r["alert"])})
    else:
        try:
            for c, (la, lo) in COORDS.items():
                f = get_forecast(la, lo, 7).copy()
                f["heat_index"] = heat_index_c(
                    f["temperature_2m_max"],
                    f["relative_humidity_2m_min"]).round(1)
                f["alert"] = f["heat_index"].apply(level)
                data[c] = f
            dates = list(data["Patna"]["time"])
            day = st.date_input("Select a date",
                                value=pd.Timestamp(dates[0]),
                                min_value=pd.Timestamp(dates[0]),
                                max_value=pd.Timestamp(dates[-1]))
            for c, (la, lo) in COORDS.items():
                r = data[c][data[c]["time"] == str(day)].iloc[0]
                rows.append({"city": c, "lat": la, "lon": lo,
                             "max temp": r["temperature_2m_max"],
                             "min humidity": r["relative_humidity_2m_min"],
                             "heat index": r["heat_index"],
                             "alert": r["alert"]})
        except Exception as ex:
            st.error(f"Could not load the forecast: {ex}")
            rows, data = [], {}

    if rows:
        m = pd.DataFrame(rows)
        m["hidden danger"] = np.where(
            (m["max temp"] < 40) & (m["heat index"] >= 41), "Yes", "No")
        m["label"] = m["city"] + ": " + m["alert"]

        st.markdown(legend_html(), unsafe_allow_html=True)
        st.caption("Hidden danger = max temperature below 40 C, but Heat "
                   "Index 41 or more.")
        fig = px.scatter_map(
            m, lat="lat", lon="lon", color="alert", text="label",
            hover_name="city",
            hover_data={"lat": False, "lon": False, "label": False,
                        "max temp": True, "heat index": True,
                        "hidden danger": True},
            color_discrete_map=COLORS, category_orders={"alert": ORDER},
            zoom=3.8, center={"lat": 23.5, "lon": 82}, height=550,
            map_style="open-street-map")
        fig.update_traces(marker_size=24, textposition="top center")
        st.plotly_chart(fig)
        st.dataframe(m[["city", "max temp", "min humidity", "heat index",
                        "alert", "hidden danger"]])

        grid, title = None, ""
        if data:
            grid = pd.DataFrame(
                {c: data[c].set_index("time")["alert"] for c in data}).T
            title = "7-day outlook for all cities (forecast)"
        elif mode == "Past date":
            start = pd.Timestamp(day)
            wk = df[(df["date"] >= start) &
                    (df["date"] <= start + pd.Timedelta(days=6))]
            grid = wk.pivot(index="city", columns="date", values="alert")
            title = ("7-day outlook for all cities "
                     "(7 days starting from the selected date)")

        if grid is not None and not grid.empty:
            grid.columns = [pd.Timestamp(x).strftime("%m-%d")
                            for x in grid.columns]

            def paint(v):
                fg = "black" if v == "Yellow" else "white"
                return (f"background-color:{COLORS.get(str(v), '#888888')};"
                        f"color:{fg}")

            st.markdown(f"### {title}")
            st.dataframe(grid.style.map(paint))
            st.caption("Columns are dates (month-day). Each cell shows the "
                       "alert level for that city on that day.")
    elif mode == "Past date":
        st.warning("No data found for this date.")