# Rent the Runway Recommendations <img width=90 align="right" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Duke_University_logo.svg/1024px-Duke_University_logo.svg.png">

## Project Abstract
Recommendation engines (RE) are widely used in the e-commerce fashion industry. However, most REs suffer from the cold start problem (where users have no prior shopping history) and the data sparsity problem (lack of user and item ratings). In this paper, we propose a novel method using a combination of unsupervised clustering, supervised learning models, and Cosine Matrix Factorization (CosMF) to address these problems, specifically in the online fashion rental industry. We found that clustering the customer base on the basis of their data inputs (i.e. body shape, size, height and weight) and then recommending items within those clusters while accounting for cold start and data sparsity with CosMF (particularly for the medium cluster) gave us the best results.

## Data Collection
### RenttheRunway Data:
- Download here: http://deepx.ucsd.edu/public/jmcauley/renttherunway/renttherunway_final_data.json.gz
- Source: https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit
### Amazon Kaggle Competition
- https://discourse.aicrowd.com/t/datasets-released-submissions-open/7522

## References
- Vincent's Talk: https://www.youtube.com/watch?v=68ABAU_V8qI
- Survey on Recommendation System: https://ieeexplore.ieee.org/abstract/document/7124857?casa_token=LelT0KsrMEIAAAAA:fIYbiIbml_C3SyQWGVXvxVn9aP2W6yd3LmNY5rmAqdkN-e_AvlR2_EBnWeS5qeHslSWOj67I
- Neural Collaborative filtering (PyTorch): https://www.youtube.com/watch?v=O4lk9Lw7lS0
- Tensorflow implementations: https://www.youtube.com/playlist?list=PLTjbQZu8nnshv8H5tV0fGh0ECBIXwuHJb
- Categorical embeddings: https://arxiv.org/pdf/1604.06737v1.pdf; https://github.com/entron/entity-embedding-rossmann


## Contributors
| Name | Reference |
|---- | ----|
|Anna Dai | [GitHub Profile](https://github.com/dai-anna)|
|Sarwari Das |[GitHub Profile](https://github.com/sarwaridas)|
|Tigran Harutyunyan |[GitHub Profile](https://github.com/HarTigran)|
|Surabhi Trivedi |[GitHub Profile](https://github.com/surabhitri)|
