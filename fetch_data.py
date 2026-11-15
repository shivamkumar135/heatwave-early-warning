import time
import requests
import pandas as pd

cities = {
    "Patna": (25.59, 85.14),
    "Gaya": (24.80, 85.00),
    "Lucknow": (26.85, 80.95),
    "Jaipur": (26.91, 75.79),
    "Guwahati": (26.14, 91.74),
    "Kolkata": (22.57, 88.36),
    "Bengaluru": (12.97, 77.59),
}

frames = []
for name, (lat, lon) in cities.items():
    print("Downloading", name, "...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2015-01-01",
        "end_date": "2025-12-31",
        "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,relative_humidity_2m_min",
        "timezone": "Asia/Kolkata",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"])
    d["city"] = name
    frames.append(d)
    time.sleep(3)

df = pd.concat(frames, ignore_index=True)
df.to_csv("raw_weather.csv", index=False)
print(df.head())
print(df.shape)