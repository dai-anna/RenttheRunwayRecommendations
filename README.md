# Rent the Runway Recommendations <img width=90 align="right" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Duke_University_logo.svg/1024px-Duke_University_logo.svg.png">

## Project Summary
Recommendation engines (RE) are widely used in the e-commerce fashion industry. However, most REs suffer from the cold start problem (where users have no prior shopping history) and the data sparsity problem (lack of user and item ratings). In this paper, we propose a novel method using a combination of unsupervised clustering, supervised learning models, and Cosine Matrix Factorization (CosMF) to address these problems, specifically in the online fashion rental industry. We built a recommendation system for Rent The Runway, an e-commerce platform that allows users to rent, subscribe, or buy designer apparel and accessories and found that clustering the customer base on the basis of their data inputs (i.e. body shape, size, height and weight) and then recommending items within those clusters while accounting for cold start and data sparsity with CosMF (particularly for the medium cluster) gave us the best results.

## Data Source
For this study, we used a Rent The Runway data with 192,544 rows of unique transactions, and 15 columns with customer and item metadata. This data can be retrieved at the https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit website.

## Experimental Design
![Experimental Design](artifacts/plots/experimental_design.png?raw=true "Title")


## Reproduce our Study
**Step 1: Clone & Navigate to the Repo:**
```
git clone https://github.com/dai-anna/RenttheRunwayRecommendations && 
cd RenttheRunwayRecommendations
```

**Step 2: Download the Source Data**
```
wget http://deepx.ucsd.edu/public/jmcauley/renttherunway/renttherunway_final_data.json.gz -P data/
```
**Step 3: Create & Activate Virtual Environment**
```
python3 -m venv venv
source venv/bin/activate
```
**Step 4: Install Dependencies**
```
pip install -r requirements.txt
```
**Step 5: Run Our Scripts**
*Note*: To run the experiment for the first time, make sure to turn all the "switches" on in all the scripts (i.e. `IWANTTOSAVEMYDATA = True`)
*Note*: We use relative file paths in the scripts
```
cd src
```
1. Clean up source data with **`preprocessing.py`**
2. Impute missing data and one hot encode with **`impute_OHE.py`**
3. Benchmark Logistic Regression, KNN, Random Forest, and XGBoost models on the general dataset with **`impute_OHE.py`**

## Contributors
| Name | Reference |
|---- | ----|
|Anna Dai | [GitHub Profile](https://github.com/dai-anna)|
|Sarwari Das |[GitHub Profile](https://github.com/sarwaridas)|
|Tigran Harutyunyan |[GitHub Profile](https://github.com/HarTigran)|
|Surabhi Trivedi |[GitHub Profile](https://github.com/surabhitri)|
