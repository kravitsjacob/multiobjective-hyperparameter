Tuning your machine learning model's hyperparameters is a great way to tailor your model's performance. Your model keeps over-fitting to your training data? Tuning your hyperparameters can help! Whenever you tune a model's hyperparameters you need to select some objective to quantify desirable model performance. For some problems, one of which will be discussed in this post, you may want to examine how model performance changes across *several* objectives. We will see that this process allows us to make a more-informed choice of hyperparameters! 

## What are Hyperparameter and Why Should We Tune Them?

The specification of hyperparameters values (whether optimal or default) is essential to many of the common machine learning algorithms because they specify model topology and ultimately model performance. For example, the number of trees in a random forest or sizes of the hidden layers in a neural network would both be be specified by hyperparameters. If you are looking for more explanation of what hyperparameters are and how they differ from regular parameters, I suggest you read [this](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/#:~:text=In\%20summary\%2C\%20model\%20parameters\%20are,be\%20set\%20manually\%20and\%20tuned) article or even just browse through the [Wikipedia Page](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) on hyperparameters.

Now I want to be clear, tuning hyperparameters is generally a "finishing touches" step when developing a machine learning model. Shahul ES said it best in [this](https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020) article by stating that hyperparameter tuning is a great way to "extract the last juice out of your models". Processes like model or feature selection cannot be overlooked. Moral of the story: if your model is severely under-performing or just flat-out broken, don't look to hyperparameter tuning to fix all your problems.

## Why Even Consider Multiple Objectives?

For many problems a single objective effectively captures desirable model performance. For example, consider the classic [Iris Classification Problem](https://archive.ics.uci.edu/ml/datasets/iris) where a model classifies types or iris flowers. You are probably thinking that for this task you want a model that is as accurate as possible, and I would agree! For the Iris problem, I would definetly pick the set of hyperparameters that maximizes model accuracy. 
 
 But now think about other classic problem of breast cancer diagnosis based on the [Breast Cancer Wisconsin dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\%28Diagnostic\%29). In this case, *how* your model classifies patients has very different implications. This is best visualized with its confusion matrix:

<img src=Figures/Cancer%20Confusion%20Matrix%20Figure.png>

In that false negative case your model is telling people they don't have cancer when the do, while in the false positive case your model is falsely scaring people by telling them they do have cancer when they don't. Each of these scenarios is undesirable but in different ways. Now one thing we could do is is optimize our hyperparameters to perform well on objectives like false positive rate or true positive rate which explicitly considers those undesirable cases. But in that case we are saying that we \textit{only} care about one of those off diagonal cases, which is often not true. Another common approach we could do is optimize to some weighted sum of false postive rate and true positive rate. But then the question becomes how do you weight the two objectives? Are they equally important? Maybe one is slightly more important? To further complicated the issue, there is a chance that the weighting scheme won't actually impact your optimal hyperparameters. 

The good news is that much smarter people than myself have thought about how solve to multi-objective problems *without* needing to weight the objectives before the optimization. To use these methods we will need to rethink what "optimality" actually means (which we will get to later). Through this methods we will be able to study the degree to which objectives tradeoff and make an informed deicion of optimal hyperparameters. Let's do an example!

## Breast Cancer Wisconsin (Diagnostic) Example

We will use the previously introduced UCI Breast Cancer Diagnostic problem with a decision tree for this example. The code I am providing will be specific to using a decision tree in Python, but the methods could easily be adapted to many hyperparameterized machine laerning algorithms in many modern programming languages. 

### Dependencies

This example will use the [Sciki-Learn](https://scikit-learn.org/stable/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Pymoo](https://pymoo.org/) libraries for the actual analysis. We will us the [HiPlot](https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/) packages to do some interactive visualization. Install them in your current environment if you haven't already done so. If you want to go the Github route, here is the repository with goodies needed to run the code I will provide! That repository also has a dockerfile if you want to go that route. 

Once you have everything installed you should be able to import everything as such. We also create some naming variables which we will leave in the global scope.

```markdown
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.tree
import sklearn.ensemble
import pymoo.util.nds.non_dominated_sorting as nds
import hiplot as hip

cv_objs = ['Mean CV Accuracy', 'Mean CV True Positive Rate', 'Mean CV False Positive Rate', 'Mean CV AUC']
cv_objs_max = ['Mean CV Accuracy', 'Mean CV True Positive Rate', 'Mean CV AUC']
test_objs = ['Test Accuracy', 'Test True Positive Rate', 'Test False Positive Rate', 'Test AUC']
```

### Data Preparation
Fortunately, the UCI Breast Cancer dataset is available for direct import via Scikit-Learn! Feature selection is the process of determining only the most important features for your problem. Feature selection won't be the focus of this post, so I encourage you to read [this](https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2) helpful blog post if you aren't familiar with feature selection. We do a basic feature selection using the feature importances from a random forest. After we do our feature selection, we split the data and save 25 % for testing our model. We do a (stratified split)[https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/] here. I have provided a simple function and function call to do this:

```markdown
def dataPreparation():
    # Import
    data = sklearn.datasets.load_breast_cancer(as_frame=True)
    features = data.feature_names.tolist()
    df = data.frame
    df['Classification'] = data['target'].replace({1: 'benign', 0: 'malignant'})
    # Feature Selection
    clf = sklearn.ensemble.RandomForestClassifier(random_state=1008)
    clf.fit(df[features], df['Classification'])
    feature_importances = pd.Series(list(clf.feature_importances_),
                                    index=features).sort_values(ascending=False)
    important_features = feature_importances[0:5].index.tolist()
    # Split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[important_features],
                                                                                df['Classification'],
                                                                                test_size=0.25,
                                                                                random_state=1008,
                                                                                stratify=df['Classification'])
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = dataPreparation()
```

### Default Hyperparameters

Let's look at the performance of the default hyperparameters. These values are typically derived based on statistical proofs and are meant to perform okay on many problems.

```markdown
def defaultHyperparameter(X_train, y_train):
    clf = sklearn.tree.DecisionTreeClassifier(random_state=1008)
    clf.fit(X_train, y_train)
    return clf
    
clf_df = defaultHyperparameter(X_train, y_train)
print(clf_df.get_params())
print('Train Accuracy:', sklearn.metrics.accuracy_score(y_train, clf_df.predict(X_train)))
print('Test Accuracy:', sklearn.metrics.accuracy_score(y_test, clf_df.predict(X_test)))
```

After running this code we see that the training accuracy is 1 and the test accuracy is just of 0.9. This means that our decision tree is being overfit to our training data. Here is a great opportunity to tune our hyperparameters so as not to overfit!

### Single Objective Hyperparameter Tuning

For this example we will focus on the two hyperparameters of a decision tree. In this single objective verseion, we want to find the set of hyperparameters that maximizes accuracy. We will specify a "grid" of possible values over which we will tune. This grid yields 84 possible combinations.

<img src=Figures/Hyperparameter%20Table.PNG>




Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/kravitsjacob/multiobjective-hyperparmater/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
