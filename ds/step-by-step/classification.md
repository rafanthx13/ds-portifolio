# Classification Snippets



## Split in Train/Test Strategies

### Check Balanced between train and test

```python
def check_balanced_train_test_binary(x_train, y_train, x_test, y_test, total_size, labels):
    """ To binary classification
    each paramethes is pandas.core.frame.DataFrame
    @total_size = len(X) before split
    @labels = labels in ordem [0,1 ...]
    """
    train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

    prop_train = train_counts_label/ len(original_ytrain)
    prop_test = test_counts_label/ len(original_ytest)
    original_size = len(x)

    print("Original Size:", '{:,d}'.format(original_size))
    print("\nTrain: must be 80% of dataset:\n", 
          "the train dataset has {:,d} rows".format(len(original_Xtrain)),
          'this is ({:.2%}) of original dataset'.format(len(original_Xtrain)/original_size),
                "\n => Classe 0 ({}):".format(labels[0]), train_counts_label[0], '({:.2%})'.format(prop_train[0]), 
                "\n => Classe 1 ({}):".format(labels[1]), train_counts_label[1], '({:.2%})'.format(prop_train[1]),
          "\n\nTest: must be 20% of dataset:\n",
          "the test dataset has {:,d} rows".format(len(original_Xtest)),
          'this is ({:.2%}) of original dataset'.format(len(original_Xtest)/original_size),
                  "\n => Classe 0 ({}):".format(labels[0]), test_counts_label[0], '({:.2%})'.format(prop_test[0]),
                  "\n => Classe 1 ({}):".format(labels[1]),test_counts_label[1], '({:.2%})'.format(prop_test[1])
         )
```

```python
# Example of use
from sklearn.model_selection import train_test_split

# split X and Y
x = df.loc[:,["Sex","Age","Category"]]
y = df.loc[:,["Survived"]]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=400)

# Using function
check_balanced_train_test_binary(x_train, y_train, x_test, y_test, len(df), ['No_Fraud', 'Fraud'])

"""
Original Size: 989

Train: must be 80% of dataset:
 train dataset have 791 rows is (79.98%) of original dataset 
 => Classe 0 (No Fraud): 679 (85.84%) 
 => Classe 1 (Fraud):    112 (14.16%) 

Test: must be 20% of dataset:
 test dataset have 198 rows is (20.02%) of original dataset 
 => Classe 0 (No Fraud) 173 (87.37%) 
 => Classe 1 (Fraud)    25 (12.63%)
"""
```

### Dividir DataSet em train/test com mesma porcentagem de classes

```python
from sklearn.model_selection import KFold, StratifiedKFold

# Split in 80% Train and 20% Test with same quantity of classes by subset
# See % of classe 0 and 1, is the same between Train and Test
kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kfold.split(x, y):
    x_train, x_test = x.iloc[train_index].values, x.iloc[test_index].values
    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

check_balanced_train_test_binary(x_train, y_train, x_test, y_test, len(df), ['Death', 'Survives'])
```

### No Balanced Split

```python
from sklearn.model_selection import train_test_split

# No Balanced (See % in classe 1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x,y, test_size=0.2, random_state=400)

check_balanced_train_test_binary(x_train1, y_train1, x_test1, y_test1, len(df), ['Death', 'Survives'])
```



## Models Devs

### Test All models

```python
# use: x_train, y_train, x_test, y_test

# Classifier Libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Ensemble Classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Others Linear Classifiers
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier

# xboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# scores
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# neural net of sklearn
from sklearn.neural_network import MLPClassifier

# others
import time
import operator

# def neural nets
mlp = MLPClassifier(verbose = False, max_iter=1000, tol = 0.000010,
                    solver = 'adam', hidden_layer_sizes=(100), activation='relu')

# def classifiers

nn_classifiers = {
    "Multi Layer Perceptron": mlp
}

linear_classifiers = {
    "SGDC": SGDClassifier(),
    "Ridge": RidgeClassifier(),
    "Perceptron": Perceptron(),
    "PassiveAggressive": PassiveAggressiveClassifier()
}

gboost_classifiers = {
    "XGBoost": XGBClassifier(),
    "LightGB": LGBMClassifier(),
}

classifiers = {
    "Naive Bayes": GaussianNB(),
    "Logisitic Regression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

ensemble_classifiers = {
    "AdaBoost": AdaBoostClassifier(),
    "GBoost": GradientBoostingClassifier(),
    "Bagging": BaggingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier()    
}

all_classifiers = {
    "Simple Models": classifiers,
    "Ensemble Models": ensemble_classifiers,
    "GBoost Models": gboost_classifiers,
    "NeuralNet Models": nn_classifiers,
    "Others Linear Models": linear_classifiers,
}

metrics = {
    'cv_scores': {},
    'acc_scores': {},
    'f1_mean_scores': {},
}

format_float = "{:.4f}"

is_print = False # True/False

time_start = time.time()

print("Fit Many Classifiers")

for key, classifiers in all_classifiers.items():
    if (is_print):
        print("\n{}\n".format(key))
    for key, classifier in classifiers.items():
        t0 = time.time()
        # xsm_train, ysm_train || x_train, y_train
        classifier.fit(x_train, y_train) 
        t1 = time.time()
        # xsm_train, ysm_train || x_train, y_train
        training_score = cross_val_score(classifier, x_train, y_train, cv=5) 
        y_pred = classifier.predict(x_test)
        cv_score = round(training_score.mean(), 4) * 100
        acc_score = accuracy_score(y_test, y_pred)
        f1_mean_score = f1_score(y_test, y_pred, average="macro") # average =  'macro' or 'weighted'
        if (is_print):
            print(key, "\n\tHas a training score of", 
                  cv_score, "% accuracy score on CrossVal with 5 cv ")
            print("\tTesting:")
            print("\tAccuracy in Test:", format_float.format(acc_score))
            print("\tF1-mean Score:", format_float.format(f1_mean_score)) 
            print("\t\tTime: The fit time took {:.2} s".format(t1 - t0), '\n')
        metrics['cv_scores'][key] = cv_score
        metrics['acc_scores'][key] = acc_score
        metrics['f1_mean_scores'][key] = f1_mean_score
        
time_end = time.time()
        
print("\nDone in {:.5} s".format(time_end - time_start), '\n')
        
print("Best cv score:", max( metrics['cv_scores'].items(), key=operator.itemgetter(1) ))
print("Best Accuracy score:", max( metrics['acc_scores'].items(), key=operator.itemgetter(1) ))
print("Best F1 score:", max( metrics['f1_mean_scores'].items(), key=operator.itemgetter(1) ))

lists = [list(metrics['cv_scores'].values()),
         list(metrics['acc_scores'].values()),
         list(metrics['f1_mean_scores'].values())
        ]

a_columns = list(metrics['cv_scores'].keys())

df_metrics = pd.DataFrame(lists , columns = a_columns,
                    index = ['cv_scores', 'acc_scores', 'f1_scores'] )
```

**Generate DataFrame with Scores**

```python
lists = [list(metrics['cv_scores'].values()),
         list(metrics['acc_scores'].values()),
         list(metrics['f1_mean_scores'].values())
        ]

a_columns = list(metrics['cv_scores'].keys())

dfre1 = pd.DataFrame(lists , columns = a_columns,
                    index = ['cv_scores', 'acc_scores', 'f1_scores'] )

dfre1 = dfre1.T.sort_values(by="acc_scores", ascending=False)
dfre1
```

## Unbalanced DataSet

### OverSampling and UnderSampling in trainDataSet

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import classification_report_imbalanced

print("train dataset before", x_train.shape[0])

SMOTE_strategy = "SMOTE" # SMOTE, SMOTEENN, SMOTETomek
print("SMOTE STRATEGY:", SMOTE_strategy)

if(SMOTE_strategy == "SMOTE"):
    sm = SMOTE('minority', random_state=42)
elif(SMOTE_strategy == "SMOTEENN"):
    sm = SMOTEENN("minority", random_state=42)
elif(SMOTE_strategy == 'SMOTETomek'):
    sm = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    
# Treina os dados originais utilizando SMOTE
xsm_train, ysm_train = sm.fit_sample(x_train, y_train)

print("train dataset before", xsm_train.shape[0],
      'generate', xsm_train.shape[0] - x_train.shape[0] )
```

## Create DF of errors of model

```python
# XGBoost
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
y_preds = xgb.predict(x_test)

# Create DataFrame of Errors of Model
df_post_pred = pd.DataFrame(
    np.concatenate((x_test, y_test, y_preds.reshape(len(y_test), 1)), axis=1),
    columns = list(x.columns) + ['Y_Target'] + ['Y_Pred']
)

df_error = df_post_pred.query("Y_Target != Y_Pred")
df_error
```

## Classification Report with ConfMatrix

````python
def class_report(y_target, y_preds, name="", labels=None):
    if(name != ''):
        print(name,"\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=labels))
````

## Save and Load Models

Even in kaggle directorys

````python
import pickle
Pkl_Filename = "Pickle_Model.pkl"

# Save the Modle to file in the current working directory
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(ensemble, file)
    
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Pickled_ensemble = pickle.load(file)

Pickled_ensemble
````

