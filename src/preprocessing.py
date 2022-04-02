# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# read in data
df = pd.read_json("../data/renttherunway_final_data.json", lines=True)

# %%
# rename columns
rename_dict = {
    "bust size": "bust_size",
    "rented for": "rented_for",
    "body type": "body_type",
    "weight": "weight_lbs",  # found it's all in lbs
}
df.rename(columns=rename_dict, inplace=True)

# %%
# quick look
df.info()
# we have missing values and need to fix some data types

###############################################################################
######################### CHECK/CLEAN EACH COLUMN #############################

# %%
# evaluate fit
df.fit.value_counts()  # OK

# %%
# fix bust size data
df.bust_size.value_counts()

# too many categories, but there are meaning in number and letter, so split into two categories
df["bust_size_num"] = df.bust_size.str.extract("(\d+)").astype(float)
df["bust_size_letter"] = df.bust_size.str.extract("(\D+)")

# rename categories according to http://www.wirarpa.com/2019/04/28/measure-bra-size/
df.bust_size_letter.replace(
    {
        "dd": "e",
        "d+": "d",
        "ddd/e": "d",
    },
    inplace=True,
)

# %%
# fix weight to numeric
df.weight_lbs = df.weight_lbs.str.replace("lbs", "").astype(float)

# %%
# fix rented_for categories
df.rented_for.value_counts()

# only one data point in "party: cocktail" so merge with "party"
df.rented_for.replace("party: cocktail", "party", inplace=True)

# %%
# check review data
df.review_text.value_counts()  # these are reviews -> should drop if not for NLP analysis
df.review_summary.value_counts()  # should also drop if not for NLP analysis

# %%
# check body_type
df.body_type.value_counts()  # OK

# %%
# check clothing categories
df.category.value_counts()

# plot distribution of categories
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(df.category, ax=ax)

# should potentially consolidate categories
df.category.unique()

#%%
# check height
df.height.value_counts()

# height is in ft and inches so split on space to inches and rename columns
df = pd.merge(
    df,
    df.height.str.split(" ", expand=True).rename(
        {0: "height_ft", 1: "height_in"}, axis=1
    ),
    left_index=True,
    right_index=True,
)

# convert ft to inches then add converted inches to inches
df.height_ft = df.height_ft.str.replace("'", "").astype(float) * 12
df.height_in = df.height_in.str.replace('"', "").astype(float) + df.height_ft

# %%
# check date
df.review_date.value_counts()

df = pd.merge(
    df,
    df.review_date.str.split(" ", expand=True).rename(
        {0: "review_month", 1: "review_day_of_month", 2: "review_year"}, axis=1
    ),
    left_index=True,
    right_index=True,
)

# change months to numbers
month_dict = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
df.review_month.replace(month_dict, inplace=True)

# convert to separated vars to int type
df.review_month = df.review_month.astype(int)
df.review_day_of_month = df.review_day_of_month.str.extract("(\d+)").astype(int)
df.review_year = df.review_year.astype(int)

# convert initial date to datetime
df["review_date"] = pd.to_datetime(df["review_date"])

# %%
# list of columns to drop
drop_cols = [
    "bust_size",
    "review_text",
    "review_summary",
    "height",
    "height_ft",
]
df.drop(columns=drop_cols, inplace=True)

# %%
# lists of columns to convert data types
cat_vars = [
    "fit",
    "bust_size_letter",
    "rented_for",
    "body_type",
]

df[cat_vars] = df[cat_vars].astype("category")

# %%
df.head()
# %%
df.info()
# %%
# save data to csv
IWANTTOUPDATEMYDATA = True
if IWANTTOUPDATEMYDATA:
    df.to_csv("../artifacts/cleandata.csv", index=False)

# %%
