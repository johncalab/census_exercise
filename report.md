# Report

The present data is survey data consisting of roughly 200k samples, with 41 demographic features. Additionally, an 'instance weight' column is present, providing weights due to stratified sampling. We wish to achieve two goals: income prediction, population segmentation. 

Let's discuss the first. The data contains a categorical feature 'label', containing two categories: "<$50k", ">$50k". We wish to train a binary classifier to predict the label of a given sample. This is a supervised learning problem.

The second goal is an unsupervised learning problem. Given the population, we wish to segment it into different groups.

## Exploration

We first look at the columns present in the data, and check their data types. Both continuous and discrete features are present.

#### Weight
Since we also have sample weight for each sample (the 'instance weight' column), we would like to know whether there is considerable difference between the raw data, and the data weighted by the sample weight. To this end, eighted. Indeed, the distributions are nearly identical.

#### Class imbalance
We notice a major class imbalance in the label features, indeed only 6% of the population has an income above $50k. This should be taken into account when evaluating the classifier.

#### Nan values
We then spend considerable time exploring the data. The first thing we notice is that there aren't many missing values. About 900 appear, and are only concentrated in the 'hispanic origin' column, which is categorical. Since the overwhelming majority falls into the 'hispanic origin_All Other' category, we fill those missing values by assigning the value 'All Other'.

#### Qualitative analysis
We plot histograms of the continuous features, divided by label, to check if any are good indicators of label. For example, the 'age' feature is a good marker for the <50k label. On the other hand, 'dividends from stocks' is a good marker for the >50k.

Another benefit of the plots is that we spot a few features which, in spite of having a numerical data type, they should really be treated as categorical (for example 'year').

For categorical features, we note that many come with a 'Not in Universe' category. This is not quite a missing value, but more an indication that the survey question was not relevant for the particular sample. An argument could be made to dropping altogether features where the NiU value is predominant, but we choose to keep them and see if the model could autonomously take care of it.

#### Correlations
We compute linear correlation values for the continuous features, but no strong correlations are detected.
We then one-hot encode the categorical features, and compute correlations. We find some very strong correlation between features, some with value 1.0, indicating duplicates (e.g. 'detailed industry recode', 'major industry code'). We then drop duplicate features from the dataset.

#### Grouping
A few categorical features come with many values, but with not much data presence. A prime example being 'country of birth self', where the vast majority is "US", some are 'Mexico', and the rest is mixed among other countries. We create a new feature 'country', grouping 'country of birth self' into three categories: "US", "Mexico", "Other". A similar approach could be done for other categorical features.

After the data cleaning process, we are left with 26 features (excluding 'label', 'instance weight' and before one-hot encoding).

#### PCA
To further reduce feature dimension, we perform PCA on the one-hot encoded variables, after applying a standard scaler. This turns out to be not very effective (or too effective, depending on the point of view). Indeed, the first two principal components explain already 99% of the variance. By plotting a scatter plot of the data projected onto the first two factors, we see that most data is concentrated near the origin. We choose not use PCA.

## Income model
After performing an 80/20 split (which was actually done before PCA), we train two different binary classifier. The first is a logistic regression with L1 regularization. To find the best regularization parameter, we perform a grid search with five-fold cross-validation, using f1 as a metric.

The second model we train is a random forest with twenty trees. The random forest performed better. This is not surprising, especially given the presence of so many categorical variables.

### Evaluation
f1, precision, recall, auc, and accuracy are all similar on train and test data, indicating that the model has not over-fit. Plotting the roc curve and the precision-recall curve, reveals that the model has good recall (once an appropriate threshold has been chosen) but poor precision.

To discuss feature importance, we use the permutation method. For both training and testing, the most important features are (roughly) the same: 'major occupation code', 'education', 'age', 'sex', 'weeks worked in year'.

## Segmentation
We now look at all the data, one-hot encoded, inclusive of the 'label' feature, and wish to segment the population. As a first approach, we try a hierarchical clustering method. This takes too long to run, so we then try k-means. To decide on the number of clusters, we use mini-batch k-means with a batch-size of 10k (5% of the samples). By plotting the obtained inertia (which is a measure of how spread the clusters are), we decide four clusters to be an optimal number (though in hindsight, eight could be a better choice). We notice a few spikes in the elbow plot, perhaps due to the randomness of the k-means initialization. Though it might be interesting to investigate.

#### Clusters
The segmentation does not produce great results. The first cluster contains most of the data, 
How the data is divided among clusters is not apparent. cluster_0 contains most of the data, which likely explains the spikes in the elbow plot (after the first group, the other clusters are random).
We plot a histogram of 'age' by cluster. We notice that cluster_0 is the only one containing samples younger than 20, while cluster_3 has the biggest concentration of elderly. Finally, we plot a scatter plot of 'age', 'capital gains', colored by cluster type. We see that cluster_1 is neatly separated from the rest of the data.


## Next steps
* To improve both models, better feature selection would be helpful, for example by doing more feature engineering.
* To improve the income prediction, more sophisticated tree-based models could be used, such as gradient boosted trees or XGBoost.
* To improve segmentation, a few different things come to mind.
    * Attempt to use more clusters in k-means.
    * Use a different clustering method, though it will take more time to train.
    * Use a rule-based + ML model for segmentation. This could be especially useful given a specific goal for the segmentation.


