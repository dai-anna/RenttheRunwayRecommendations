# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# %%
# read in data
df = pd.read_json("../data/renttherunway_final_data.json", lines=True)

# %%
# rename columns
def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {
        "bust size": "bust_size",
        "rented for": "rented_for",
        "body type": "body_type",
        "weight": "weight_lbs",  # found it's all in lbs
    }

    df.rename(columns=rename_dict, inplace=True)

    return df


rename_cols(df)

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


def split_bust_size(df: pd.DataFrame) -> pd.DataFrame:
    # too many categories, but there are meaning in number and letter, so split into two categories
    df["bust_size_num"] = df.bust_size.str.extract("(\d+)")
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

    return df


# %%
# check weights
df.weight_lbs.value_counts()

# %%
# fix rented_for categories
df.rented_for.value_counts()

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


def consolidate_categories(df: pd.DataFrame) -> pd.DataFrame:
    # only one data point in "party: cocktail" so merge with "party"
    df.rented_for.replace("party: cocktail", "party", inplace=True)
    return df


#%%
# check height
df.height.value_counts()


def combine_height(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


# %%
# check date
df.review_date.value_counts()


def split_dates(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


# %%
# drop extra columns
def drop_extra_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "bust_size",
        "review_text",
        "review_summary",
        "height",
        "height_ft",
    ]

    df.drop(columns=drop_cols, inplace=True)

    return df


# %%
# fix data types
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    cat_vars = [
        "fit",
        "bust_size_letter",
        "rented_for",
        "body_type",
    ]

    df[cat_vars] = df[cat_vars].astype("category")

    num_vars = [
        "bust_size_num",
        "weight_lbs",
        "review_month",
        "review_day_of_month",
        "review_year",
    ]

    for var in num_vars:
        df[var] = df[var].astype(str).str.extract("(\d+)")
        try:
            df[var] = df[var].astype(int)
        except:
            df[var] = df[var].astype(float)

    df["review_date"] = pd.to_datetime(df["review_date"])

    return df


# %%
# add response variable
def add_response_var(df: pd.DataFrame) -> pd.DataFrame:
    df["recommend"] = df.rating == 10
    df["recommend"] = df.recommend.astype(int)

    return df


# %%
clean_df = (
    df.copy()
    .pipe(split_bust_size)
    .pipe(consolidate_categories)
    .pipe(combine_height)
    .pipe(split_dates)
    .pipe(drop_extra_cols)
    .pipe(convert_data_types)
    .pipe(add_response_var)
)

# %%
clean_df.head()
# %%
clean_df.info()

# %%
# train-test-validation split
X_train, X_val_test, y_train, y_val_test = train_test_split(
    clean_df.copy().drop(columns=["recommend"]),
    clean_df["recommend"],
    test_size=0.4,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test,
    y_val_test,
    test_size=0.5,
    random_state=42,
)

# %%
# alternatively if we want to implement k-fold cross validation
# https://stackoverflow.com/questions/39748660/how-to-perform-k-fold-cross-validation-with-tensorflow

# %%
# save data to disk
IWANTTORESAVEMYDATA = True

if IWANTTORESAVEMYDATA:
    clean_df.to_parquet("../artifacts/cleandata.parquet")
    pd.concat([X_train, y_train], axis=1).to_parquet("../artifacts/train.parquet")
    pd.concat([X_val, y_val], axis=1).to_parquet("../artifacts/val.parquet")
    pd.concat([X_test, y_test], axis=1).to_parquet("../artifacts/test.parquet")

# %%
# try find users who only have one item in their history
users_items = df.groupby("user_id").count()["item_id"].sort_values(ascending=True)
# find user ids that only have more than one item in their history
user_ids_morethan1 = users_items[users_items > 1].index
reduced_df = df[df["user_id"].isin(user_ids_morethan1)]

if IWANTTORESAVEMYDATA:
    reduced_df.to_parquet("../artifacts/reduceddata.parquet")

# %%
