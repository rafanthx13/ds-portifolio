# REGRESSION

## CrossValidation KFold Regression Models

```python
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, scale
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LassoCV, BayesianRidge, LassoLarsIC
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.svm import SVR

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
```

```python
# Setup cross validation folds
kf = KFold(n_splits=4, random_state=42, shuffle=True)

# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X_train):
    rmse = np.sqrt(-cross_val_score(model, X, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
```

```python
# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# setup models    
lasso_alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

elastic_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
elastic_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# Lasso Regressor
lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=lasso_alphas2,
                              random_state=42, cv=kf))
# Elastic Net Regressor
elasticnet = make_pipeline(RobustScaler(),  
                           ElasticNetCV(max_iter=1e7, alphas=elastic_alphas,
                                        cv=kf, l1_ratio=elastic_l1ratio))

# Kernel Ridge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)
```

```python
regressor_models = {
    'LightGB': lightgbm, # 20s
    'XGBoost': xgboost, # 340s = 5min 40s
    'SVM_Regressor': svr, # 6s
    'Ridge': ridge, # 6s
    'RandomForest': rf, # 146s = 2min 20s
    'GradientBoosting': gbr, # 93s = 1min 30s
    # 'stack_gen': stack_gen, # Nâo tem como fazer, esse CV é para avaliar os outros, nao tem como aplicar o CV ao Stack
    ## ADD++
    'Lasso': lasso, # 15s
    'KernelRidge': KRR, # 1.88s
    'ElasticNet': elasticnet # 40s
}

scores = {}

## Cross Validation
t_start = time.time()

for model_name, model in regressor_models.items():
    print(model_name)
    t0 = time.time()
    score = cv_rmse(model)
    t1 = time.time()
    m, s = score.mean(), score.std()
    scores[model_name] = [m,s]
    print('\t=> mean {:.5f}, std: {:.5f}'.format(m, s))
    print("\t=> took {:,.3f} s".format(t1 - t0))
    
t_ending = time.time()
print('took', t_ending - t_start)
```

## Fit Many Regression Models with Time

```python
# train a model and show the time
# Def Stack Model: Stack up all the models above, optimized using  ['xgboost'/'elasticnet']
stack_gen = StackingCVRegressor(regressors = (xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor = elasticnet,
                                use_features_in_secondary=True)

# train a model and show the time
def fit_a_model(model, model_name):
    t0 = time.time()
    if(model_name == 'Stack'):
        a_model = model.fit( np.array(X_train), np.array(y_train) )
    else:
        a_model =  model.fit( X_train, y_train )
    t1 = time.time()
    print("{} took {:,.3f} s".format(model_name, t1 - t0))
    return a_model

lgb_model   = fit_a_model(lightgbm, 'LightGB') # 3.7s
svr_model   = fit_a_model(svr, 'SVM_R') # 1.8s
ridge_model = fit_a_model(ridge, 'Ridge') # 1.8s
gbr_model   = fit_a_model(gbr, 'GradientBoost') # 30s
lasso_model = fit_a_model(lasso, 'Lasso') # 2.8s
kridg_model = fit_a_model(KRR, 'KernelRidge') # 1.2
elast_model = fit_a_model(elasticnet, 'ElasticNet') # 8s
# more time
rf_model    = fit_a_model(rf, 'RandomForest') # 51s
xgb_model   = fit_a_model(xgboost, 'XGboost') # 116s = 2min
stack_model = fit_a_model(stack_gen, 'Stack') # 1.087s = 18min
```

