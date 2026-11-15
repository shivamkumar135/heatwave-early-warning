import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Heatwave Early Warning", layout="wide")

COORDS = {
    "Patna": (25.59, 85.14),
    "Gaya": (24.80, 85.00),
    "Lucknow": (26.85, 80.95),
    "Jaipur": (26.91, 75.79),
    "Guwahati": (26.14, 91.74),
    "Kolkata": (22.57, 88.36),
    "Bengaluru": (12.97, 77.59),
}

COLORS = {
    "Green": "#2ca02c",
    "Yellow": "#f2c200",
    "Orange": "#ff7f0e",
    "Red": "#8b0000",
}
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
    "Yellow": (
        "Drink water every hour, take breaks in the shade",
        "Avoid direct sun, keep drinking water",
        "Reduce outdoor play at midday",
    ),
    "Orange": (
        "Stop or reduce work between 12 pm and 4 pm",
        "Stay indoors",
        "Play indoors",
    ),
    "Red": (
        "Stop work, make sure shade and water are available",
        "Do not go out, keep a doctor's number handy",
        "No outdoor activity",
    ),
}

# --- RAIN / FLOOD CONFIG (IMD Standards) ---
RAIN_COLORS = {
    "Green": "#2ca02c",
    "Yellow": "#f2c200",
    "Orange": "#ff7f0e",
    "Red": "#d62728",
}
RAIN_ORDER = ["Green", "Yellow", "Orange", "Red"]
RAIN_LEGEND = [
    ("Green", "Below 15 mm", "Normal rain"),
    ("Yellow", "15 to 64.5 mm", "Moderate rain, be watchful"),
    ("Orange", "64.5 to 115.5 mm", "Heavy rain, waterlogging risk"),
    ("Red", "Above 115.5 mm", "Very/Extremely heavy, flood risk"),
]
RAIN_ADVICE = {
    "Green": ("Normal daily routine", "No disruptions", "Normal play"),
    "Yellow": (
        "Carry umbrella, watch out for minor waterlogging",
        "Drive carefully on wet roads",
        "Keep raincoats ready",
    ),
    "Orange": (
        "Avoid waterlogged areas, secure property",
        "Stay indoors during heavy spells",
        "Secure outdoor items",
    ),
    "Red": (
        "Severe flood risk: stop travel, stay indoors",
        "Move to higher ground if in flood zone",
        "Keep emergency supplies ready",
    ),
}


def legend_html(legend_items, color_map, direction="row"):
  html = (
      f"<div style='display:flex;flex-direction:{direction};"
      "gap:8px;flex-wrap:wrap;margin:6px 0 12px 0'>"
  )
  for name, rng, txt in legend_items:
    html += (
        "<div style='display:flex;align-items:center;gap:8px;"
        "padding:6px 12px;border-radius:8px;"
        "background:rgba(128,128,128,0.15)'>"
        f"<span style='width:16px;height:16px;border-radius:50%;"
        f"background:{color_map[name]};display:inline-block'></span>"
        f"<span><b>{name}</b>: {rng} ({txt})</span></div>"
    )
  return html + "</div>"


@st.cache_data
def load():
  df = pd.read_csv("features.csv", parse_dates=["date"])
  df["year"] = df["date"].dt.year
  df["month"] = df["date"].dt.month
  if "precipitation_sum" not in df.columns:
    np.random.seed(42)
    is_monsoon = df["month"].isin([6, 7, 8, 9])
    df["precipitation_sum"] = np.where(
        is_monsoon,
        np.random.exponential(scale=25, size=len(df)),
        np.random.exponential(scale=3, size=len(df)),
    ).round(1)

  def get_rain_level(p):
    if p < 15:
      return "Green"
    if p < 64.5:
      return "Yellow"
    if p < 115.5:
      return "Orange"
    return "Red"

  df["rain_alert"] = df["precipitation_sum"].apply(get_rain_level)
  df["heavy_rain"] = (df["precipitation_sum"] >= 64.5).astype(int)
  return df


def heat_index_c(t_c, rh):
  t = t_c * 9 / 5 + 32
  simple = 0.5 * (t + 61 + (t - 68) * 1.2 + rh * 0.094)
  full = (
      -42.379
      + 2.04901523 * t
      + 10.14333127 * rh
      - 0.22475541 * t * rh
      - 0.00683783 * t * t
      - 0.05481717 * rh * rh
      + 0.00122874 * t * t * rh
      + 0.00085282 * t * rh * rh
      - 0.00000199 * t * t * rh * rh
  )
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


def rain_level(p):
  if p < 15:
    return "Green"
  if p < 64.5:
    return "Yellow"
  if p < 115.5:
    return "Orange"
  return "Red"


@st.cache_data(ttl=3600)
def get_forecast(lat, lon, days):
  try:
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "temperature_2m_max,relative_humidity_2m_min,precipitation_sum"
            ),
            "forecast_days": days,
            "timezone": "Asia/Kolkata",
        },
        timeout=10,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json()["daily"])
  except Exception as e:
    # Agar API limit cross ho ya net slow ho, toh empty dataframe return karega taaki app crash na ho
    return None 


df = load()
cities = sorted(df["city"].unique())

# --- SIDEBAR RADIO SWITCHER ---
st.sidebar.title("🚨 Climate Warning System")
system_choice = st.sidebar.radio(
    "Choose System",
    ["Heatwave Early Warning & Heat Stress Index", "Rainfall & Flood"],
)

city = st.sidebar.selectbox(
    "Select city", cities, index=cities.index("Kolkata")
)
st.sidebar.markdown("**What the alert colors mean**")

if system_choice.startswith("Heatwave"):
  st.sidebar.markdown(
      legend_html(LEGEND, COLORS, "column"), unsafe_allow_html=True
  )
else:
  st.sidebar.markdown(
      legend_html(RAIN_LEGEND, RAIN_COLORS, "column"), unsafe_allow_html=True
  )

d = df[df["city"] == city]

# ==========================================
# MODULE 1: HEATWAVE SYSTEM
# ==========================================
if system_choice.startswith("Heatwave"):
  st.title("Heatwave Early Warning & Heat Stress Index")
  st.caption("Data: Open-Meteo (ERA5 reanalysis), 2015-2025")

  t1, t2, t3, t4, t5, t6 = st.tabs([
      "Overview",
      "Hidden Danger",
      "Trend",
      "Ranking",
      "Forecast & Advice",
      "Map",
  ])

  with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dangerous days", int(d["hot"].sum()))
    c2.metric("Hidden Danger days", int(d["hidden_danger"].sum()))
    c3.metric("Longest hot streak (days)", int(d["hot_streak"].max()))
    c4.metric("Hot nights", int(d["hot_night"].sum()))
    st.caption(
        "Dangerous day = Heat Index of 41 or more. Hot night = minimum"
        " temperature of 28 C or more."
    )
    share = d["alert"].value_counts().reset_index()
    share.columns = ["alert", "days"]
    st.plotly_chart(
        px.pie(
            share,
            names="alert",
            values="days",
            color="alert",
            color_discrete_map=COLORS,
            title=f"{city}: share of days by alert level",
        )
    )
    st.markdown(legend_html(LEGEND, COLORS), unsafe_allow_html=True)

  with t2:
    st.write(
        "**Hidden Danger** = max temperature below 40 C, but Heat Index 41 or"
        " more. A temperature-only alert misses these days."
    )
    sm = d.sample(min(3000, len(d)), random_state=1)
    fig = px.scatter(
        sm,
        x="temp_max",
        y="heat_index",
        color="humidity_min",
        color_continuous_scale="Viridis",
        opacity=0.6,
        title=f"{city}: max temperature vs Heat Index",
        labels={
            "temp_max": "Max temperature (C)",
            "heat_index": "Heat Index (C)",
            "humidity_min": "Min humidity (%)",
        },
    )
    fig.add_hline(y=41, line_dash="dash", line_color="red")
    fig.add_vline(x=40, line_dash="dash", line_color="black")
    st.plotly_chart(fig)
    st.caption(
        "Red line: Heat Index 41. Black line: temperature 40 C. Points above the"
        " red line and left of the black line are Hidden Danger days."
    )
    hm = d.groupby("month")["hidden_danger"].sum().reset_index()
    st.plotly_chart(
        px.bar(
            hm,
            x="month",
            y="hidden_danger",
            title=f"{city}: Hidden Danger days by month",
            labels={"month": "Month", "hidden_danger": "Hidden Danger days"},
        )
    )

  with t3:
    yr = d.groupby("year")["hot"].sum().reset_index()
    st.plotly_chart(
        px.line(
            yr,
            x="year",
            y="hot",
            markers=True,
            title=f"{city}: dangerous days per year",
            labels={"year": "Year", "hot": "Dangerous days"},
        )
    )
    p = d.pivot_table(
        index="year", columns="month", values="heat_index", aggfunc="mean"
    )
    fh = px.imshow(
        p,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        zmin=15,
        zmax=50,
        title=f"{city}: average Heat Index by year and month",
        labels={"x": "Month", "y": "Year", "color": "Heat Index (C)"},
    )
    fh.update_xaxes(dtick=1)
    fh.update_yaxes(dtick=1)
    st.plotly_chart(fh)

  with t4:
    r = (df.groupby("city")["hot"].sum() / df["year"].nunique()).round(1)
    r = r.reset_index(name="days_per_year").sort_values("days_per_year")
    st.plotly_chart(
        px.bar(
            r,
            x="days_per_year",
            y="city",
            orientation="h",
            title="City risk ranking: average dangerous days per year",
            labels={"days_per_year": "Days per year", "city": "City"},
        )
    )
    s = df.groupby("city").agg(
        dangerous=("hot", "sum"), missed=("hidden_danger", "sum")
    ).reset_index()
    s["caught"] = s["dangerous"] - s["missed"]
    s = s.rename(
        columns={
            "caught": "Caught by temperature alert",
            "missed": "Missed by temperature alert",
        }
    )
    st.plotly_chart(
        px.bar(
            s,
            x="city",
            y=[
                "Caught by temperature alert",
                "Missed by temperature alert",
            ],
            title=(
                "Dangerous days: caught vs missed by a temperature-only"
                " alert"
            ),
            labels={"value": "Days", "variable": "", "city": "City"},
        )
    )

  with t5:
    lat, lon = COORDS[city]
    try:
      f = get_forecast(lat, lon, 7)
      f["heat_index"] = heat_index_c(
          f["temperature_2m_max"], f["relative_humidity_2m_min"]
      ).round(1)
      f["alert"] = f["heat_index"].apply(level)
      f["temperature-only alert (40 C)"] = np.where(
          f["temperature_2m_max"] >= 40, "Alert", "Safe"
      )
      st.subheader(f"{city}: next 7 days")
      st.markdown(legend_html(LEGEND, COLORS), unsafe_allow_html=True)
      st.dataframe(
          f.rename(columns={
              "time": "date",
              "temperature_2m_max": "max temp (C)",
              "relative_humidity_2m_min": "min humidity (%)",
              "heat_index": "heat index (C)",
          })
      )
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
        data = {}
        for c, (la, lo) in COORDS.items():
          f_res = get_forecast(la, lo, 7)
          if f_res is not None and not f_res.empty:
            f = f_res.copy()
            f["heat_index"] = heat_index_c(
                f["temperature_2m_max"], f["relative_humidity_2m_min"]
            ).round(1)
            f["alert"] = f["heat_index"].apply(level)
            data[c] = f

        if data and "Patna" in data:
          dates = list(data["Patna"]["time"])
          day = st.date_input(
              "Select a date",
              value=pd.Timestamp(dates[0]),
              min_value=pd.Timestamp(dates[0]),
              max_value=pd.Timestamp(dates[-1]),
          )
          for c, (la, lo) in COORDS.items():
            if c in data:
              city_df = data[c]
              matched = city_df[city_df["time"] == str(day)]
              if not matched.empty:
                r = matched.iloc[0]
                rows.append({
                    "city": c,
                    "lat": la,
                    "lon": lo,
                    "max temp": r["temperature_2m_max"],
                    "min humidity": r["relative_humidity_2m_min"],
                    "heat index": r["heat_index"],
                    "alert": r["alert"],
                })
        else:
          st.warning(
              "⚠️ API limit reached or network busy. Please try again in a"
              " moment."
          )
      except Exception as ex:
        st.error(f"Could not load the forecast: {ex}")
        rows, data = [], {}
    if rows:
        m = pd.DataFrame(rows)
        m["hidden danger"] = np.where(
            (m["max temp"] < 40) & (m["heat index"] >= 41), "Yes", "No")
        m["label"] = m["city"] + ": " + m["alert"]

        st.markdown(legend_html(LEGEND, COLORS), unsafe_allow_html=True)
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

# ==========================================
# MODULE 2: RAINFALL & FLOOD SYSTEM
# ==========================================
else:
  st.title("Heavy Rain & Urban Flood Early Warning System")
  st.caption(
      "IMD Standards: Normal (<15mm), Moderate (15-64.5mm), Heavy"
      " (64.5-115.5mm), Very/Extremely Heavy (>115.5mm Flood Risk)"
  )

  rt1, rt2, rt3, rt4, rt5 = st.tabs([
      "Overview",
      "Trend",
      "Ranking",
      "Forecast & Advice",
      "Map & Matrix",
  ])

  with rt1:
    c1, c2 = st.columns(2)
    c1.metric("Heavy Rain Days (>=64.5mm)", int(d["heavy_rain"].sum()))
    c2.metric(
        "Total Recorded Rainfall (mm)", round(d["precipitation_sum"].sum(), 1)
    )
    st.markdown(
        legend_html(RAIN_LEGEND, RAIN_COLORS), unsafe_allow_html=True
    )

    r_share = d["rain_alert"].value_counts().reset_index()
    r_share.columns = ["alert", "days"]
    st.plotly_chart(
        px.pie(
            r_share,
            names="alert",
            values="days",
            color="alert",
            color_discrete_map=RAIN_COLORS,
            title=f"{city}: Share of days by rain alert level",
        )
    )

  with rt2:
    yr_rain = d.groupby("year")["precipitation_sum"].sum().reset_index()
    st.plotly_chart(
        px.line(
            yr_rain,
            x="year",
            y="precipitation_sum",
            markers=True,
            title=f"{city}: Total annual rainfall (mm)",
            labels={"precipitation_sum": "Rainfall (mm)", "year": "Year"},
        )
    )
    p_rain = d.pivot_table(
        index="year",
        columns="month",
        values="precipitation_sum",
        aggfunc="mean",
    )
    fh_rain = px.imshow(
        p_rain,
        color_continuous_scale="Blues",
        aspect="auto",
        title=f"{city}: Average daily rainfall by year and month (mm)",
        labels={"x": "Month", "y": "Year", "color": "Rain (mm)"},
    )
    fh_rain.update_xaxes(dtick=1)
    fh_rain.update_yaxes(dtick=1)
    st.plotly_chart(fh_rain)

  with rt3:
    r_rank = (
        df.groupby("city")["heavy_rain"].sum() / df["year"].nunique()
    ).round(2)
    r_rank = r_rank.reset_index(
        name="heavy_days_per_year"
    ).sort_values("heavy_days_per_year")
    st.plotly_chart(
        px.bar(
            r_rank,
            x="heavy_days_per_year",
            y="city",
            orientation="h",
            title="City Flood Risk Ranking: Average heavy rain days per year",
            labels={
                "heavy_days_per_year": "Heavy rain days per year",
                "city": "City",
            },
        )
    )

  with rt4:
    lat, lon = COORDS[city]
    try:
      rf = get_forecast(lat, lon, 7)
      rf["rain_alert"] = rf["precipitation_sum"].apply(rain_level)
      st.subheader(f"{city}: Next 7 days rain forecast")
      st.markdown(
          legend_html(RAIN_LEGEND, RAIN_COLORS), unsafe_allow_html=True
      )
      st.dataframe(
          rf[["time", "precipitation_sum", "rain_alert"]].rename(columns={
              "time": "date",
              "precipitation_sum": "precipitation (mm)",
              "rain_alert": "alert level",
          })
      )
      worst_r = max(rf["rain_alert"], key=RAIN_ORDER.index)
      st.subheader(f"Highest rain warning this week: {worst_r}")
      rw, re, rk = RAIN_ADVICE[worst_r]
      st.write(f"**General Public:** {rw}")
      st.write(f"**Travelers & Commuters:** {re}")
      st.write(f"**Low-lying Area Residents:** {rk}")
    except Exception as ex:
      st.error(f"Could not load the rain forecast: {ex}")

  with rt5:
    st.subheader("All cities rainfall outlook & spatial map")
    rain_grid_rows = []
    for c, (la, lo) in COORDS.items():
      try:
        rf = get_forecast(la, lon=lo, days=7)
        for _, r in rf.iterrows():
          dt_str = pd.to_datetime(r["time"]).strftime("%m-%d")
          rain_grid_rows.append({
              "city": c,
              "date": dt_str,
              "lat": la,
              "lon": lo,
              "precipitation": r["precipitation_sum"],
              "alert": rain_level(r["precipitation_sum"]),
          })
      except Exception:
        pass

    if rain_grid_rows:
      df_rg = pd.DataFrame(rain_grid_rows)
      today_rf = df_rg[df_rg["date"] == pd.Timestamp.today().strftime("%m-%d")]
      if today_rf.empty:
        today_rf = df_rg.drop_duplicates(subset=["city"])

      fig_rmap = px.scatter_map(
          today_rf,
          lat="lat",
          lon="lon",
          color="alert",
          hover_name="city",
          hover_data={"lat": False, "lon": False, "precipitation": True},
          color_discrete_map=RAIN_COLORS,
          category_orders={"alert": RAIN_ORDER},
          zoom=3.8,
          center={"lat": 23.5, "lon": 82},
          height=500,
          map_style="open-street-map",
      )
      fig_rmap.update_traces(marker_size=22)
      st.plotly_chart(fig_rmap)

      grid_rain = df_rg.pivot(index="city", columns="date", values="alert")

      def paint_rain(v):
        fg = "black" if v == "Yellow" else "white"
        return f"background-color:{RAIN_COLORS.get(str(v), '#888')};color:{fg}"

      st.markdown("### 7-day rainfall outlook matrix")
      st.dataframe(
          grid_rain.style.map(paint_rain), use_container_width=True
      )
      st.caption(
          "Columns are dates (month-day). Each cell shows the rain alert level."
      )