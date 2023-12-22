<h1>Instructions given to us</h1>

There are two datasets (Data1 and Data2) in this project. For each dataset, we need to develop a solution to predict the class labels of the testing records in the dataset. A record in the datasets contains several attributes. The real-world meaning of the class labels and attributes are removed and will be explained after the deadline, which will let us sense the power of big data analytics to generate accurate results without expert knowledge of a certain application domain.

<h1>Problem definition</h1>

A classification task was conducted on two distinct datasets, which prediction of class label on the testing records from these two datasets was performed.

<h1>General approach</h1>
The whole implementation was written in R language. 

<h2>Dataset 1</h2>
XGBoost is tree based ensemble model, which involves gradient boosting including gradient boosted trees, regularization, approximate greedy algorithm, weighted quantile sketch, sparsity-aware split finding, parallel learning, cache-aware access and blocks of out-of-core computation (Tianqi Chen & Carlos Guestrin, 2016).  

Prior to inputting the training records into the model, feature engineering was used to ensure that only specific columns were fed into the model. In this case, topic modeling was performed first before using XGBoost model.

Topic modelling, a text mining method, was performed on the text data in the dataset. The algorithm in use was Latent Dirichlet allocation (LDA). LDA treats “each document as a mixture of topics, and each topic as a mixture of words, which allows documents to “overlap” each other in terms of content, rather than being separated into discrete groups, in a way that mirrors typical use of natural language” (Julia Silge & David Robinson, 2017). Unlike Principal Component Analysis (PCA), every document is a mixture of topics and every topic is a mixture of words for LDA.

Since class imbalance happened among the training record, Synthetic Minority Oversampling Technique (SMOTE) was also used to address this issue.

<h2>Dataset 2</h2>
Random Forest, another tree-based ensemble model, amalgamates the output of multiple decision trees to arrive at a single result, which this process involves the construction of numerous trees by the algorithm before the predictions are averaged (Breiman, 2001).

Before building Random Forest model, Predictive Mean Matching (PMM) was used for missing data handling. PMM is a common imputation method for estimating the values when data are missing at random which was the assumption for the missing data on this dataset. 
