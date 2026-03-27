import pandas as pd

"""
This will require the combination with the air quality data to allow an in_leaf column
"""

# Load the CSV - Just makwe sure to edit with your directory
df = pd.read_csv(r"C:\Users\emyan\Applied Data Science\ADS_group\Tree_data_processing\final_tree_dataset_potential.csv")

top_species = df["COMMON_NAME"].value_counts().head(22).index

print("\nMost common tree species:", top_species)

names = df["FULL_COMMON_NAME"].dropna()

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

df[["leaf_start", "leaf_end"]] = df["COMMON_NAME"].apply(
    lambda x: pd.Series(get_leaf_period(x))
)

def is_in_leaf(month, start, end):
    return 1 if start <= month <= end else 0

# print(df)