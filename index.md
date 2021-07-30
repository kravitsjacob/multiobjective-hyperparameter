Tuning your machine learning model's hyperparameters is a great way to tailor your model's performance. Your model keeps overfitting to your training data? Tuning your hyperparameters can help! When you tune a model's hyperparameters, you need to select some objective to quantify desirable model performance. For some problems, one of which will be discussed in this post, you may want to examine how model performance changes across *several* objectives. We will see that this process allows us to make a more informed choice of hyperparameters! 

## What are Hyperparameter and Why Should We Tune Them?

The specification of hyperparameter values (whether optimal or default) is essential to many common machine learning algorithms because they specify model topology and ultimately model performance. For example, the number of trees in a random forest or sizes of the hidden layers in a neural network would both be specified by hyperparameters. If you are looking for more explanation of what hyperparameters are and how they differ from regular parameters, I suggest you read [this website](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/#:~:text=In\%20summary\%2C\%20model\%20parameters\%20are,be\%20set\%20manually\%20and\%20tuned), [this article](https://www.jmlr.org/papers/volume20/18-444/18-444.pdf), or even just browsing through the [Wikipedia page](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) on hyperparameters.

I want to be clear, tuning hyperparameters is generally a "finishing touches" step when developing a machine learning model. Shahul ES said it well in [this blog post](https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020) by stating that hyperparameter tuning is a great way to "extract the last juice out of your models." Processes like model or feature selection cannot be overlooked. Moral of the story: if your model is severely under-performing or just flat-out broken, don't look to hyperparameter tuning to fix all your problems.

## Why Even Consider Multiple Objectives?

For many problems a single objective effectively captures desirable model performance. For example, consider the classic [Iris classification problem](https://archive.ics.uci.edu/ml/datasets/iris) where a model classifies types of Iris flowers. You are probably thinking that for this task you want a model that is as accurate as possible, and I would agree! For the Iris problem, I would pick the set of hyperparameters that maximizes model accuracy. 
 
But now think about the other classic problem of breast cancer diagnosis based on the [Breast Cancer Wisconsin dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\%28Diagnostic\%29) currently hosted in the University of California, Irvine repository. In this case, *how* your model classifies patients has very different implications. This is best visualized with its confusion matrix:

<img src=Figures/Cancer%20Confusion%20Matrix%20Figure.png />

In the false negative case, your model is telling people they *don't* have cancer when they *do*. In the false positive case, your model is falsely scaring people by telling them they *do* have cancer when they *don't*. Each of these scenarios is undesirable but in different ways. One thing we could do is optimize our hyperparameters to perform well on objectives like false positive rate or true positive rate which explicitly considers those undesirable cases. But in those cases, we are saying that we *only* care about one of those off diagonal cases, which is often not true. Another common approach we could do is optimize to some weighted sum of false positive rate and true positive rate. But then the question becomes how do you weight the two objectives? Are they equally important? Maybe one is slightly more important? To further complicate the issue, there is a chance that some weighting schemes will not impact the actual values of optimal hyperparameters. 

The good news is that much smarter people than me have thought about how to solve to multi-objective problems *without* needing to weight the objectives before the optimization. To use these methods, we will need to rethink what "optimality" actually means (which we will get to later). Through these methods we will be able to study the degree to which objectives tradeoff and make an informed decision of optimal hyperparameters. Let's do an example!

## Breast Cancer Wisconsin (Diagnostic) Example

We will use the previously introduced UCI Breast Cancer Diagnostic problem with a decision tree for this example. The code I am providing will be specific to using a decision tree in Python, but the methods could easily be adapted to many hyperparameterized machine learning algorithms in many modern programming languages. 

#### Dependencies

This example will use the [Sciki-Learn](https://scikit-learn.org/stable/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Pymoo](https://pymoo.org/) libraries for the actual analysis. We will us the [HiPlot](https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/) packages to do some interactive visualization. Install them in your current environment if you haven't already done so. If you want to go the Github route, [here is the repository]( https://github.com/kravitsjacob/multiobjective-hyperparameter) which contains the script as well as goodies needed to run the code I will provide! Specifically, that repository has a dockerfile and virtual environment dependencies. I recommend using one of those two options to ensure consistent results with the blog post (I have also included instructions for those not familiar with either method). However, the code should run in any environment with Python 3 and all the proper dependencies installed.

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

#### Data Preparation
Fortunately, the UCI Breast Cancer dataset is available for direct import via Scikit-Learn (so no need to manually download it yourself)! Feature selection is the process of determining only the most important features for your problem. Feature selection won't be the focus of this post, so I encourage you to read [this blog post](https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2) or [this article](https://www.aaai.org/Papers/Symposia/Fall/1994/FS-94-02/FS94-02-034.pdf) if you aren't familiar with feature selection. We do a basic feature selection using the feature importances from a random forest. After we do our feature selection, we split the data and save 25 % for testing our model. We do a [stratified split](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/). I have provided a simple function and function call to do this:

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

#### Default Hyperparameters

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

After running this code, we see that the training accuracy is 1.00 and the test accuracy is 0.91. This means that our decision tree is being overfit to our training data. Here is a great opportunity to tune our hyperparameters so as not to overfit!

#### Single Objective Hyperparameter Tuning

For this example, we will focus on two hyperparameters of a decision tree. In this single objective version, we want to find the set of hyperparameters that maximizes accuracy. We will specify a "grid" of possible values over which we will tune. This grid yields 84 possible combinations.

<img src=Figures/Hyperparameter%20Table.PNG>

Additionally, we will be evaluating performance using five-fold cross validation. In this technique, we iteratively split our training data as to not overfit our model to the entire training data set as was done in the previous section. For more information on what cross validation means [here is a blog post](https://medium.com/@mandava807/cross-validation-and-hyperparameter-tuning-in-python-65cfb80ee485) and [here is an article]( https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full). 

This analysis is applied in the following code:

```markdown
def singleObjectiveGridSearch(X_train, X_test, y_train, y_test):
    parameter_grid = {'min_samples_split': np.insert(np.arange(10, 210, 10), 0, 2), 'max_features': [2, 3, 4, 5]}
    gs = sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeClassifier(random_state=1008),
                                              parameter_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    clf = sklearn.tree.DecisionTreeClassifier(min_samples_split=gs.best_params_['min_samples_split'],
                                 max_features=gs.best_params_['max_features'], random_state=1008)
    clf.fit(X_train, y_train)
    return clf, gs
    
clf_SO, gs_SO = singleObjectiveGridSearch(X_train, X_test, y_train, y_test)
print(clf_SO.get_params())
print('CV Train Accuracy:', gs_SO.best_score_)
print('Test Accuracy:', sklearn.metrics.accuracy_score(y_test, gs_SO.predict(X_test)))
```

Running this code yields that our cross-validated training accuracy has dropped to 0.94 (from 1.00) but our accuracy of predicting the test set has increased to 0.94 (from 0.91). This is great news; our model is no longer being overfit to our data! 

But let's return to our discussion about multiple objectives. Maximizing accuracy is sort of maximizing the "greatest good" which very much falls in line with the [philosophy of Jeremy Bentham](https://en.wikipedia.org/wiki/Utilitarianism). What about other objectives like false positive rate or true positive rate (linked [here](https://en.wikipedia.org/wiki/Confusion_matrix) which consider that minority of people that our model misclassifies? How do we consider those objectives without having to rank or weight them?

#### Multi-Objective Hyperparameter Tuning

In this multi-objective formulation, we will study the tradeoffs among the accuracy, false positive rate, true positive rate, and area under receiver operator characteristic curve objectives. So, we start out by computing each of those five objectives for our 84 hyperparameter combinations in our grid. Just as in the single objective case, these objectives are evaluated in a five-fold cross-validated fashion on the training set.

```markdown
def fpr(y_true, y_pred):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    obj = fp / (fp + tn)
    return obj


def tpr(y_true, y_pred):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    obj = tp / (tp + fn)
    return obj


def multiObjectiveGridSearch(X_train, y_train):
    parameter_grid = {'min_samples_split': np.insert(np.arange(10, 210, 10), 0, 2),
                      'max_features': [2, 3, 4, 5]}
    scoring = {'Accuracy': 'accuracy', 'True Positive Rate': sklearn.metrics.make_scorer(tpr),
               'False Positive Rate': sklearn.metrics.make_scorer(fpr), 'AUC': 'roc_auc'}
    gs = sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeClassifier(random_state=1008),
                                              parameter_grid, cv=5, scoring=scoring, n_jobs=-1, refit=False)
    gs.fit(X_train, y_train)
    df = pd.DataFrame(gs.cv_results_['params'])
    df['Mean CV Accuracy'] = gs.cv_results_['mean_test_Accuracy']
    df['Mean CV True Positive Rate'] = gs.cv_results_['mean_test_True Positive Rate']
    df['Mean CV False Positive Rate'] = gs.cv_results_['mean_test_False Positive Rate']
    df['Mean CV AUC'] = gs.cv_results_['mean_test_AUC']
    return df
    
    
df_all = multiObjectiveGridSearch(X_train, y_train)
```

In this code you will notice that we defined our own true positive rate and false positive rate functions while the other two objectives are built in to scikit-learn. I wanted to show how easy it is to extend scikit-learn's functionality!

How can we visualize the objective performance of our 84 hyperparameter combinations? I'm a big fan of interactive parallel plots which can be easily implemented via the HiPlot package although [many similar tools exist in other languages](https://github.com/ParasolJS):

```markdown
def parallelPlot(df, color_column, invert_column):
    exp = hip.Experiment.from_dataframe(df)
    exp.parameters_definition[color_column].colormap = 'interpolateViridis'
    exp.display_data(hip.Displays.PARALLEL_PLOT).update({'hide': ['uid', 'max_features', 'min_samples_split'],
                                                         'invert': invert_column})
    exp.display_data(hip.Displays.TABLE).update({'hide': ['from_uid']})
    return exp


parallelPlot(df_all, color_column='Mean CV Accuracy', invert_column=cv_objs_max).to_html('all.html')
```

INSERT HTML HERE

So what are we looking at here? Each hyperparameter combination is represented as a single line on this plot (you can see this as you hover over the table on the bottom). We oriented the axes such that down is optimal, meaning a solution that performed best on all objectives would be a straight line across the bottom. However, we don't see any solutions with that behavior instead we see the objectives trading off performance with one another. 

But let's study two solutions in this plot to compare their performance: solution with uid 66 (```max_features```: 5, ```min_sample_split```: 30) and solution with uid 4 (```max_features```: 2, ```min_sample_split```: 40). We see that solution 66 does better on *every* objective than solution 4. By that reasoning, there would never be a reason to pick solution 4 if all we cared about were these four objectives. Commonly, we say that solution 66 "dominates" solution 4. Continuing this logic, we only really care about the nondominated sets of hyperparameters. So we apply a nondominated sort to this set and re-plot:

```markdown
def nondomSort(df, objs, max_objs=None):
    df_sorting = df.copy()
    # Flip Objectives to Maximize
    if max_objs is not None:
        df_sorting[max_objs] = -1.0 * df_sorting[max_objs]
    # Nondominated Sorting
    nondom_idx = nds.find_non_dominated(df_sorting[objs].values)
    return df.iloc[nondom_idx]

df_non_dom = nondomSort(df_all, cv_objs, max_objs=cv_objs_max)
parallelPlot(df_non_dom, color_column='Mean CV Accuracy', invert_column=cv_objs_max).to_html('non_dom.html')
```

INSERT HTML HERE

We see that our original 84 solutions got filtered out to just seven non-dominated solutions! I want to be clear that *all seven* of these solutions are "optimal" which may be a bit hard to wrap your head around if you are new to multi-objective optimization. Another way to think about it: Imagine you tried every combination of objective weights for these four objectives, you would always get one of these seven non-dominated hyperparameter solutions. This concept is also called [Pareto optimality](https://en.wikipedia.org/wiki/Pareto_efficiency#:~:text=Pareto\%20efficiency\%20or\%20Pareto\%20optimality,or\%20without\%20any\%20loss\%20thereof.) if you want to read further.

We can gain some insights into this problem through our plot of the set of non-dominated hyperparameters above! We can see that accuracy and false positive rate are generally redundant objectives. This means that by maximizing accuracy we are also reducing the amount of people we incorrectly diagnosing who truly have cancer. We see that accuracy generally conflicts with true positive rate (i.e., you can't increase performance in one objective without decreasing performance in the other). Recall that this means that by maximizing accuracy our model is falsely scaring people by telling them they have cancer when they don't. What is nice about this multi-objective approach is that we can visually see to what extents these objectives tradeoff. We can also pick a solution, like solution 2, that compromises among all the objectives. There are also [analytical methods for selecting a compromise solution]( https://pubs.acs.org/doi/10.1021/acs.iecr.6b03453) which I won’t cover in this post. 

Do these objective preferences translate to the “test” set we omitted at the start of this exercise? For example, does a solution that have good cross-validated accuracy also have a good accuracy on the test set? We can check this by ranking the test performance for each objective and compare the test performance to the cross-validated performance. This is simply done in the following code:

```markdown
# Non-Dominated Set Test Performance
df_non_dom_test = df_non_dom.apply(lambda row: getTestPerformance(X_train, X_test, y_train, y_test, row), axis=1)
df_non_dom = df_non_dom.join(df_non_dom_test)
# Check if Objective Performance is Preserved by Looking at Sorted Objective Values
for i, j in zip(cv_objs, test_objs):
    print(df_non_dom[[i, j]].sort_values(i, ascending=False))
```

We see that the objective preferences generally translate to the test set (there are a few exceptions due to rounding). Of course, this behavior is expected given that we used a cross-validated approach to get the non-dominated set in the first place, but it's nice to see the cross-validation is working. This is great news as we can be confident the objective preferences that we spent so much time forming and investigating throughout this process translate to new observations!

## Conclusion

There you have it! A nice way to use a grid search to conduct a simple, yet informative, multi-objective approach to tuning hyperparameters. We demonstrated how to use the default hyperparameters, how to conduct a single-objective grid search, and how to conduct a multi-objective grid search! Along the way we also showed some interactive plotting methods to view our results and make a choice of hyperparameters. Hopefully, I have inspired you to implement a similar approach for your own machine learning problem! 

[Dr. Joseph Kasprzyk](https://www.colorado.edu/lab/krg/), [Dr. Kyri Baker](http://www.kyrib.com/), [Dr. Kostas Andreadis](https://hydro-umass.github.io/), and I actually applied this methodology to the problem of dam hazard classification, another problem where the types misclassification have different consequences. We optimized over many hyperparameters (and some other parameters of a geospatial model). So, we utilized a multi-objective evolutionary algorithm to solve our multi-objective problem instead of the methods used in this post. The code for that project, a video presentation, and the paper are all available [here](https://osf.io/vyzh8/) if you are interested! 

