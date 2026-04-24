import pandas as pd
import numpy as np

"""
This file will merge the hourly air quality data with the tree
data to allow for the sigmoid 0-1 multiplier to be used for
calculating the effective crown volume
"""

# Use you own path to run this file
trees = pd.read_csv(r"C:\Users\emyan\Applied Data Science\ADS_group\Tree_data_processing\trees_within_radius.csv")
air = pd.read_csv(r"C:\Users\emyan\Applied Data Science\ADS_group\Traffic Data\continuous_air_quality_2324.csv")
sun = pd.read_csv(r"C:\Users\emyan\Applied Data Science\ADS_group\sunrise and sunset data\bristol_sun_2023.csv")

# Some trees seem to have overlapping sensors so I'm splitting them up into duplicates
trees["SENSOR_ID"] = trees["SENSOR_ID"].astype(str).str.replace(" ", "", regex=False)
trees["SENSOR_ID"] = trees["SENSOR_ID"].str.split(",")
trees = trees.explode("SENSOR_ID")
trees["SENSOR_ID"] = trees["SENSOR_ID"].str.strip()
trees = trees[trees["SENSOR_ID"] != ""]

trees["SENSOR_ID"] = trees["SENSOR_ID"].astype(str)
air["SITE_ID"] = air["SITE_ID"].astype(str)

# Tree seasonality (taken from Tree_species.py)
trees["COMMON_NAME"] = trees["COMMON_NAME"].astype(str).str.strip()
evergreens = {
    "Bay", "Box", "Cabbage Palm", "Cedar", "Chusan Palm", "Conifer", "Cypress", "Douglas Fir", "Eleagnus",
    "Eucalyptus", "Feijoa", "Fir", "Firethorn", "Hiba", "Hemlock", "Holly", "Juniper", "Laurel", "Monkey Puzzle",
    "Pine", "Pittosporum", "Redwood", "Scots Pine", "Spruce", "Strawberry Tree", "Wedding Cake Tree", "Yew"
}

leaf_periods = {
    "Lime": (4, 10), "Cherry": (4, 9), "Maple": (5, 10), "Ash": (5, 10),
    "Sycamore": (5, 10), "Plane": (5, 10), "Oak": (5, 11), "Silver Birch": (4, 9),
    "Hawthorn": (4, 10), "Apple": (5, 10), "Field Maple": (5, 10), "Horse Chestnut": (4, 9),
    "Alder": (4, 10), "Hornbeam": (4, 11), "Whitebeam": (5, 10), "Rowan": (5, 9),
    "Poplar": (4, 9), "Beech": (5, 11), "Willow": (4, 10), "Plum": (4, 10)
}

default_deciduous_period = (4, 10)

def get_leaf_period(name):
    if name in evergreens:
        return (1, 12)
    return leaf_periods.get(name, default_deciduous_period)

trees[["leaf_start_month", "leaf_end_month"]] = trees["COMMON_NAME"].apply(
    lambda x: pd.Series(get_leaf_period(x))
)

trees["is_evergreen"] = trees["COMMON_NAME"].isin(evergreens)

"""
I do not have the crown volume calculation so you gusy 
can add that here once you've got it
"""

# Taking the datetime from the air data and merging it with tree data
air["DATE_TIME"] = pd.to_datetime(air["DATE_TIME"], errors="coerce")
air = air.dropna(subset=["DATE_TIME"])
air["day_of_year"] = air["DATE_TIME"].dt.dayofyear
air["month"] = air["DATE_TIME"].dt.month
air["hour"] = air["DATE_TIME"].dt.hour
air["date"] = air["DATE_TIME"].dt.normalize()

# Sun data
sun["date"] = pd.to_datetime(sun["date"], errors="coerce")

sun["Sunrise_clean"] = (
    sun["Sunrise/Sunset Sunrise Sunrise"]
    .astype(str)
    .str.extract(r"(\d{1,2}:\d{2}\s*[ap]m)", expand=False)
    .str.upper()
)
sun["Sunset_clean"] = (
    sun["Sunrise/Sunset Sunset Sunset"]
    .astype(str)
    .str.extract(r"(\d{1,2}:\d{2}\s*[ap]m)", expand=False)
    .str.upper()
)

sun["sunrise_dt"] = pd.to_datetime(
    sun["date"].dt.strftime("%Y-%m-%d") + " " + sun["Sunrise_clean"],
    format="%Y-%m-%d %I:%M %p",
    errors="coerce"
)

sun["sunset_dt"] = pd.to_datetime(
    sun["date"].dt.strftime("%Y-%m-%d") + " " + sun["Sunset_clean"],
    format="%Y-%m-%d %I:%M %p",
    errors="coerce"
)

air = air.merge(
    sun[["date", "sunrise_dt", "sunset_dt"]],
    on="date",
    how="left"
)

air["hour_start"] = air["DATE_TIME"]
air["hour_end"] = air["DATE_TIME"] + pd.Timedelta(hours=1)

def daylight_fraction(row):
    start = row["hour_start"]
    end = row["hour_end"]
    sunrise = row["sunrise_dt"]
    sunset = row["sunset_dt"]

    # Completely dark
    if end <= sunrise or start >= sunset:
        return 0.0
    # Fully daylight
    if start >= sunrise and end <= sunset:
        return 1.0
    # Sunrise hour
    if start < sunrise < end:
        return (end - sunrise).total_seconds() / 3600
    # Sunset hour
    if start < sunset < end:
        return (sunset - start).total_seconds() / 3600

    return 0.0

air["sun_multiplier"] = air.apply(daylight_fraction, axis=1)

merged = air.merge(
    trees,
    left_on="SITE_ID",
    right_on="SENSOR_ID",
    how="left"
)

# Multiplier. Using days instead of months 1-365 to make it continuous
month_mid_doy = {
    1: 15, 2: 46, 3: 74, 4: 105, 5: 135, 6: 166,
    7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def leaf_multiplier(day_of_year, start_month, end_month, evergreen=False, ramp_days=30):
    if pd.isna(start_month) or pd.isna(end_month):
        return np.nan

    if evergreen:
        return 1.0

    start_doy = month_mid_doy[int(start_month)]
    end_doy = month_mid_doy[int(end_month)]
    steepness = ramp_days / 6
    up = sigmoid((day_of_year - start_doy) / steepness)
    down = 1 - sigmoid((day_of_year - end_doy) / steepness)

    return float(up * down)

merged["leaf_multiplier"] = merged.apply(
    lambda row: leaf_multiplier(
        day_of_year=row["day_of_year"],
        start_month=row["leaf_start_month"],
        end_month=row["leaf_end_month"],
        evergreen=row["is_evergreen"],
        ramp_days=30
    ),
    axis=1
)

"""
Effective crown volume calculation, can be uncommented once we have it.
After this the summed tree effect can be appllied to each sensor per hour.
Lmk if you want me to do that after, it can be added once we have the crown volume.
Name the csv file whatever but please make sure it works because I cant check it.
"""
# merged["effective_crown_volume"] = merged["crown_volume"] * merged["leaf_multiplier"]
# merged["effective_daylight_crown_volume"] =  merged["crown_volume"] * merged["leaf_multiplier"] * merged["sun_multiplier"]
# merged.to_csv(r"")
