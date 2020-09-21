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
# Unbalanced, balance, imbalance
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

**Import Libs**

```python
# Classifier Libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Ensemble Classifiers
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Others Linear Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier

# xboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# scores
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score

# neural net of sklearn
from sklearn.neural_network import MLPClassifier

# others
import time
import operator
```

**Create Models**

```python
# def neural nets
mlp = MLPClassifier(verbose = False, max_iter=1000, tol = 0.000010,
                    solver = 'adam', hidden_layer_sizes=(100), activation='relu')

all_classifiers = {
    "NaiveBayes": GaussianNB(),
    "Ridge": RidgeClassifier(),
    "Perceptron": Perceptron(),
    'NeuralNet': mlp,
    "PassiveAggr": PassiveAggressiveClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGB": LGBMClassifier(),
    "SVM": SVC(),
    "LogisiticR": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SGDC": SGDClassifier(),
    "GBoost": GradientBoostingClassifier(),
    "Bagging": BaggingClassifier(),
    "RandomForest": RandomForestClassifier(),
    "ExtraTree": ExtraTreesClassifier()
}
```

**Cv, fit and test**

```python
metrics = { 'cv_acc': {}, 'acc_test': {}, 'f1_test': {} }
m = list(metrics.keys())
time_start = time.time()
print('CrossValidation, Fitting and Testing')

m = list(metrics.keys())

print('CrossValidation, Fitting and Testing')

time_start = time.time()

# Cross Validation, Fit and Test
for name, model in all_classifiers.items():
    print('{:15}'.format(name), end='')
    t0 = time.time()
    # Cross Validation
    training_score = cross_val_score(model, x_train, y_train, scoring="accuracy", cv=4)
    # Fitting
    all_classifiers[name] = model.fit(x_train, y_train) 
    # Testing
    y_pred = all_classifiers[name].predict(x_test)
    t1 = time.time()
    # Save metrics
    metrics[m[0]][name] = training_score.mean()
    metrics[m[1]][name] = accuracy_score(y_test, y_pred)
    metrics[m[2]][name] = f1_score(y_test, y_pred, average="macro") 
    # Show metrics
    print('| {}: {:6,.4f} | {}: {:6,.4f} | {}: {:6.4f} | took: {:>15} |'.format(
        m[0], metrics[m[0]][name], m[1], metrics[m[1]][name], m[2], metrics[m[2]][name], time_spent(t0) ))
```

**Generate DataFrame with Scores**

```python
print("\nDone in {}".format(time_spent(time_start)), '\n')
print("Best cv acc  :", max( metrics[m[0]].items(), key=operator.itemgetter(1) ))
print("Best acc test:", max( metrics[m[1]].items(), key=operator.itemgetter(1) ))
print("Best f1 test :", max( metrics[m[2]].items(), key=operator.itemgetter(1) ))

df_metrics = pd.DataFrame(data = [list(metrics[m[0]].values()),
                                  list(metrics[m[1]].values()),
                                  list(metrics[m[2]].values())],
                          index = ['cv_acc', 'acc_test', 'f1_test'],
                          columns = metrics[m[0]].keys() ).T.sort_values(by=m[0], ascending=False)
df_metrics
```



## Unbalanced DataSet

### OverSampling and UnderSampling in trainDataSet

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek # over and under sampling
from imblearn.metrics import classification_report_imbalanced

imb_models = {
    'ADASYN': ADASYN(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'SMOTEENN': SMOTEENN("minority", random_state=42),
    'SMOTETomek': SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=42),
    'RandomUnderSampler': RandomUnderSampler(random_state=42)
}

imb_strategy = "RandomUnderSampler"

if(imb_strategy != "None"):
    before = x_train.shape[0]

    imb_tranformer = imb_models[imb_strategy]
    
    x_train, y_train = imb_tranformer.fit_sample(x_train, y_train)

    print("train dataset before: {:,d}\nimbalanced_strategy: {}".format(before, imb_strategy),
          "\ntrain dataset after: {:,d}\ngenerate: {:,d}".format(x_train.shape[0], x_train.shape[0] - before))

else:
    print("Dont correct unbalanced dataset")
# check_balanced_train_test_binary(x_train, y_train, x_test, y_test, len(df), ['Response 0', 'Response 1'])
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

## Plot history NeuralNet acc and loss in training

````python
def plot_nn_loss_acc(history):
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

    # summarize history for accuracy
    axis1.plot(history.history['accuracy'], label='Train', linewidth=3)
    axis1.plot(history.history['val_accuracy'], label='Validation', linewidth=3)
    axis1.set_title('Model accuracy', fontsize=16)
    axis1.set_ylabel('accuracy')
    axis1.set_xlabel('epoch')
    axis1.legend(loc='upper left')

    # summarize history for loss
    axis2.plot(history.history['loss'], label='Train', linewidth=3)
    axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)
    axis2.set_title('Model loss', fontsize=16)
    axis2.set_ylabel('loss')
    axis2.set_xlabel('epoch')
    axis2.legend(loc='upper right')
    plt.show()
````

## Classification Report with ConfMatrix

````python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score

this_labels = ['HAM','SPAM']
scoress = {}

def class_report(y_real, y_my_preds, name="", labels=this_labels):
    if(name != ''):
        print(name,"\n")
    print(confusion_matrix(y_real, y_my_preds), '\n')
    print(classification_report(y_real, y_my_preds, target_names=labels))
    scoress[name] = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')]
    
# Create DataFrame from Scores maded by 'class_report'
def create_df_fom_scores(scores=scoress):
    return pd.DataFrame(data= scoress.values(),
            columns=['acc','f1'],
            index=scoress.keys())

# Create: create_df_fom_scores()
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

