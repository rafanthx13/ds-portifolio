## Hyper Tuning Random Forest 

Links:
+ https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
+ https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
+ https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
+ https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6

### Why Use CV

The technique of cross validation (CV) is best explained by example using the most common method, K-Fold CV. When we approach a machine learning problem, we make sure to split our data into a training and a testing set. In K-Fold CV, we further split our training set into K number of subsets, called folds. We then iteratively fit the model K times, each time training the data on K-1 of the folds and evaluating on the Kth fold (called the validation data). As an example, consider fitting a model with K = 5. The first iteration we train on the first four folds and evaluate on the fifth. The second time we train on the first, second, third, and fifth fold and evaluate on the fourth. We repeat this procedure 3 more times, each time evaluating on a different fold. At the very end of training, we average the performance on each of the folds to come up with final validation metrics for the model.

![](https://miro.medium.com/max/700/0*KH3dnbGNcmyV_ODL.png)

#### Strategie GridSearch

#### Strategie: RandomSearchCV

#### Strategie: Param-to-Param

### How Start a Random Forest Classifier



````python
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier()
rf1.get_params()

{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': None,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False
}
````

Where

+ `bootstrap`:: bool, default=True
  - Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
+ `ccp_alpha`:: non-negative float, default=0.0
  - Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.
+ `class_weight`:: {“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
  - Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
+ `criterion`:: {“gini”, “entropy”}, default=”gini”
  - The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.
+ `max_depth`: int, default=None
  - The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
+ `max_features`:: {“auto”, “sqrt”, “log2”}, int or float, default=”auto”
  - The number of features to consider when looking for the best split:
    * If int, then consider max_features features at each split.
    * If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
    * If “auto”, then max_features=sqrt(n_features).
    * If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
    * If “log2”, then max_features=log2(n_features).
    * If None, then max_features=n_features.
+ `max_leaf_nodes`:: int, default=None
  - Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
+ `max_samples`:: If bootstrap is True, the number of samples to draw from X to train each base estimator.
  - If None (default), then draw X.shape[0] samples.
    * If int, then draw max_samples samples.
    * If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0, 1).
+ `min_impurity_decrease`:: float, default=0.0
  - A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
+ `min_impurity_split`: None,
+ `min_samples_leaf`:: int or float, default=1
  - The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
    * If int, then consider min_samples_leaf as the minimum number.
    * If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
+ `min_samples_split`:: int or float, default=2
  - The minimum number of samples required to split an internal node:
    * If int, then consider min_samples_split as the minimum number.
    * If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
+ `min_weight_fraction_leaf`:: float, default=0.0
  - The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
+ `n_estimators`: int, default=100
  - The number of trees in the forest.
+ `n_jobs`:: int, default=None
  - The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
+ `oob_score`:: bool, default=False
  - Whether to use out-of-bag samples to estimate the generalization accuracy.
+ `random_state`:: int or RandomState, default=None
  - Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node
+ `verbose`:: int, default=0
  - Controls the verbosity when fitting and predicting.
+ `warm_start`:: bool, default=False
  - When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.

### More important params to hyperturing

**What is HyperParam?**
Most generally, a hyperparameter is a parameter of the model that is set prior to the start of the learning process. Different models have different hyperparameters that can be set. For a Random Forest Classifier, there are several different hyperparameters that can be adjusted. In this post, I will be investigating the following four parameters:


**n_estimators**
+ number of trees in the foreset
+ n_estimators represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data. However, adding a lot of trees can slow down the training process considerably, therefore we do a parameter search to find the sweet spot.
+ The n_estimators parameter specifies the number of trees in the forest of the model. The default value for this parameter is 10, which means that 10 different decision trees will be constructed in the random forest.

**max_depth**
+ max number of levels in each decision tree
+ max_depth represents the depth of each tree in the forest. The deeper the tree, the more splits it has and it captures more information about the data. We fit each decision tree with depths ranging from 1 to 32 and plot the training and test errors.
+ The max_depth parameter specifies the maximum depth of each tree. The default value for max_depth is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class.

**min_samples_split**
+ min number of data points placed in a node before the node is split
+ min_samples_split represents the minimum number of samples required to split an internal node. This can vary between considering at least one sample at each node to considering all of the samples at each node. When we increase this parameter, each tree in the forest becomes more constrained as it has to consider more samples at each node. Here we will vary the parameter from 10% to 100% of the samples
+ The min_samples_split parameter specifies the minimum number of samples required to split an internal leaf node. The default value for this parameter is 2, which means that an internal node must have at least two samples before it can be split to have a more specific classification.

**min_samples_leaf**
+ min number of data points allowed in a leaf node
+ min_samples_leaf is The minimum number of samples required to be at a leaf node. This parameter is similar to min_samples_splits, however, this describe the minimum number of samples of samples at the leafs, the base of the tree.
+ The min_samples_leaf parameter specifies the minimum number of samples required to be at a leaf node. The default value for this parameter is 1, which means that every leaf must have at least 1 sample that it classifies.

**max_features**
+ max number of features considered for splitting a node
+ max_features represents the number of features to consider when looking for the best split.
