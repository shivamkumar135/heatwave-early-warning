import numpy as np
import pandas as pd


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


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)


print("\n--- Test 1: formula vs reference values ---")
# first two: US National Weather Service heat index chart
# third: Patna 15 Jun 2024 from world-weather.info (RealFeel)
for tf, rh, exp_f in [(90, 50, 94.6), (100, 40, 109.3), (104, 47, 127.0)]:
    got_f = float(heat_index_c((tf - 32) * 5 / 9, rh)) * 9 / 5 + 32
    check(f"{tf}F at {rh}% humidity: expected ~{exp_f}F, got {got_f:.1f}F",
          abs(got_f - exp_f) < 0.5)

print("\n--- Test 2: alert level boundaries ---")
cases = {31.9: "Green", 32: "Yellow", 40.9: "Yellow",
         41: "Orange", 53.9: "Orange", 54: "Red"}
for hi, exp in cases.items():
    check(f"Heat Index {hi} -> {exp}", level(hi) == exp)

print("\n--- Test 3: Hidden Danger rule ---")
for t, hi, exp in [(38, 45, True), (42, 44, False), (30, 33, False)]:
    check(f"temp {t}, heat index {hi} -> hidden danger = {exp}",
          ((t < 40) and (hi >= 41)) == exp)

print("\n--- Test 4: data sanity ---")
df = pd.read_csv("features.csv", parse_dates=["date"])
num = ["temp_max", "temp_min", "humidity", "humidity_min", "heat_index"]
check("no missing values", df[num].isna().sum().sum() == 0)
check("no duplicate city-date rows", not df.duplicated(["city", "date"]).any())
check("every city has the same number of days",
      df.groupby("city").size().nunique() == 1)
check("humidity is between 0 and 100",
      df["humidity_min"].between(0, 100).all())
check("max temp is never below min temp",
      (df["temp_max"] >= df["temp_min"]).all())
check("hidden danger days are always dangerous days",
      (~df["hidden_danger"] | (df["hot"] == 1)).all())

print("\n--- Test 5: our alert vs an IMD-style temperature rule ---")
df["doy"] = df["date"].dt.dayofyear
df["normal"] = df.groupby(["city", "doy"])["temp_max"].transform("mean")
imd = (((df["temp_max"] >= 40) & (df["temp_max"] - df["normal"] >= 4.5))
       | (df["temp_max"] >= 45))
ours = df["hot"] == 1
t5 = pd.DataFrame({"city": df["city"], "both": imd & ours,
                   "only IMD-style": imd & ~ours, "only ours": ~imd & ours})
print(t5.groupby("city").sum().astype(int))

print("\n--- Test 6: how much do our choices change the result? ---")
years = df["date"].dt.year.nunique()
df["hi_mean"] = heat_index_c(df["temp_max"], df["humidity"])
rows = []
for city, g in df.groupby("city"):
    rows.append({
        "city": city,
        "min-humid, HI>=39": (g["heat_index"] >= 39).sum() / years,
        "min-humid, HI>=41": (g["heat_index"] >= 41).sum() / years,
        "min-humid, HI>=43": (g["heat_index"] >= 43).sum() / years,
        "mean-humid, HI>=41": (g["hi_mean"] >= 41).sum() / years})
print("Dangerous days per year:")
print(pd.DataFrame(rows).round(1).to_string(index=False))