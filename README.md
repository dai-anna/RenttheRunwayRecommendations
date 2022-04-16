# Rent the Runway Recommendations <img width=90 align="right" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Duke_University_logo.svg/1024px-Duke_University_logo.svg.png">

## Project Summary
Recommendation engines (RE) are widely used in the e-commerce fashion industry. However, most REs suffer from the cold start problem (where users have no prior shopping history) and the data sparsity problem (lack of user and item ratings). In this paper, we propose a novel method using a combination of unsupervised clustering, supervised learning models, and Cosine Matrix Factorization (CosMF) to address these problems, specifically in the online fashion rental industry. We built a recommendation system for Rent The Runway, an e-commerce platform that allows users to rent, subscribe, or buy designer apparel and accessories and found that clustering the customer base on the basis of their data inputs (i.e. body shape, size, height and weight) and then recommending items within those clusters while accounting for cold start and data sparsity with CosMF (particularly for the medium cluster) gave us the best results.

## Data Source
For this study, we used a Rent The Runway data with 192,544 rows of unique transactions, and 15 columns with customer and item metadata. This data can be retrieved at the https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit website.

## Experimental Design
![Experimental_Design](artifacts/plots/experimental_design.png?raw=true "Experimental Design")


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
3. Benchmark Logistic Regression, KNN, Random Forest, and XGBoost models on the general dataset with **`generalmodel.py`**
4. Hyperparameter tune and train the general XGBoost model with: **`xgb_all.py`**
5. Cluster the users with: **`clustering.ipynb`**
4. Hyperparameter tune and train the cluster-specific XGBoost models with: **`xgb_clusters.py`**
5. Run our CosMF neural network model with: **`cosine_similarity_NN.py`**

## Evaluate Results
### Clusters
We determined from the dendrogram below (left figure) that 3 clusters is most appropriate for our users and visualized our clusters with the PCA-reduced diagram below (right figure):

![Dendrogram](artifacts/plots/dendrogram.png?raw=true "Dendrogram")  |  ![PCA_Clusters](artifacts/plots/PCA.png?raw=true "PCA")
|---- | ----|

Further examination of mean values within our clusters unveiled that the largest differentials are within in the `size` and `weight` features. Thus, we can label our clusters as follows:

<div align="center">
  
| Cluster | Size | Weight | Label |
|:----:|----|----|----|
| 0 | 30.34 | 192.86 | large (L) |
| 1 | 16.16 | 149.90 | medium (M) |
| 2 | 6.86 | 123.20 | small (S) |

</div>

### XGBoost Predictions
We employ supervised model to predict a user's preferences to make initial recommendations. We fine tuned hyperparameters for the XGBoost model and found that the cluster-specific models outperforms the general model in all three clusters. and it performs particularly well in the `medium` cluster:

![XGB](artifacts/plots/xgb_rocpr.png?raw=true "XGB")

### CosMF
We then use cluster-specific neural network matrix factorization to predict additional items once a user expresses their interest in any item. Our model is able to learn user- and item-embeddings within each cluster to make the best predictions.

Sample of item recommendations:

<div align="center">

|  | Item ID | Category | Rented For | Body Type 
|----|----|----|----|----|
| Reference Item | 123793 | Gown | Formal Affair | Hourglass |
| Recommendation 1 | 131533 | Gown | Formal Affair | Hourglass |
| Recommendation 2 | 130727 | Dress | Party | Hourglass |
| Recommendation 3 | 132738 | Gown | Formal Affair | Hourglass |
| Recommendation 4 | 136110 | Dress | Wedding | Hourglass |

</div>

## Final Deliverables

Our final deliverables are in the `report/` folder or see below for direct links:

- Our presentation on our preliminary results are [here](https://youtu.be/PzAVR38oM6Y)
- Our full report [here](https://github.com/dai-anna/RenttheRunwayRecommendations/raw/main/report/RTRRecommendationsFinalReport.pdf)

## Contributors

<div align="center">

| Name | Reference |
|---- | ----|
|Anna Dai | [GitHub Profile](https://github.com/dai-anna)|
|Sarwari Das |[GitHub Profile](https://github.com/sarwaridas)|
|Tigran Harutyunyan |[GitHub Profile](https://github.com/HarTigran)|
|Surabhi Trivedi |[GitHub Profile](https://github.com/surabhitri)|
  
</div>

