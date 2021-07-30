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

def dataPreparation():
    # Test Parameters
    test_size = 0.25
    number_features = 5
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
    important_features = feature_importances[0:number_features].index.tolist()
    # Split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[important_features],
                                                                                df['Classification'],
                                                                                test_size=test_size,
                                                                                random_state=1008,
                                                                                stratify=df['Classification'])
    return X_train, X_test, y_train, y_test


def defaultHyperparameter(X_train, y_train):
    clf = sklearn.tree.DecisionTreeClassifier(random_state=1008)
    clf.fit(X_train, y_train)
    return clf


def singleObjectiveGridSearch(X_train, y_train):
    parameter_grid = {'min_samples_split': np.insert(np.arange(10, 210, 10), 0, 2), 'max_features': [2, 3, 4, 5]}
    gs = sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeClassifier(random_state=1008),
                                              parameter_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    clf = sklearn.tree.DecisionTreeClassifier(min_samples_split=gs.best_params_['min_samples_split'],
                                 max_features=gs.best_params_['max_features'], random_state=1008)
    clf.fit(X_train, y_train)
    return clf, gs


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


def nondomSort(df, objs, max_objs=None):
    df_sorting = df.copy()
    # Flip Objectives to Maximize
    if max_objs is not None:
        df_sorting[max_objs] = -1.0 * df_sorting[max_objs]
    # Non-dominated Sorting
    nondom_idx = nds.find_non_dominated(df_sorting[objs].values)
    return df.iloc[nondom_idx]


def parallelPlot(df, color_column, invert_column):
    # Make Unique IDs
    df['Solution ID'] = df.index + 1
    df['Solution ID'] = df['Solution ID'].apply(lambda x: '{0:0>5}'.format(x))
    df['Solution ID'] = 'S'+df['Solution ID'].astype(str)
    # Create Plot
    exp = hip.Experiment.from_dataframe(df)
    exp.parameters_definition[color_column].colormap = 'interpolateViridis'
    exp.display_data(hip.Displays.PARALLEL_PLOT).update({'hide': ['uid', 'max_features', 'min_samples_split'],
                                                         'invert': invert_column})
    exp.display_data(hip.Displays.TABLE).update({'hide': ['uid', 'from_uid']})
    return exp


def getTestPerformance(X_train, X_test, y_train, y_test, params):
    # Fit Model with Specified Hyperparameters
    clf = sklearn.tree.DecisionTreeClassifier(min_samples_split=int(params['min_samples_split']),
                                              max_features=int(params['max_features']), random_state=1008)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Compute Objectives on Test Set
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_pred).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    auc = sklearn.metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return pd.Series([acc, tpr, fpr, auc], test_objs)


def main():
    # Prepare
    X_train, X_test, y_train, y_test = dataPreparation()
    # Default Hyperparameter Values
    clf_def = defaultHyperparameter(X_train, y_train)
    print(clf_def.get_params())
    print('Train Accuracy:', sklearn.metrics.accuracy_score(y_train, clf_def.predict(X_train)))
    print('Test Accuracy:', sklearn.metrics.accuracy_score(y_test, clf_def.predict(X_test)))
    # Single Objective Grid Search
    clf_SO, gs_SO = singleObjectiveGridSearch(X_train, y_train)
    print(clf_SO.get_params())
    print('CV Train Accuracy:', gs_SO.best_score_)
    print('Test Accuracy:', sklearn.metrics.accuracy_score(y_test, gs_SO.predict(X_test)))
    # Multi-Objective Grid Search
    df_all = multiObjectiveGridSearch(X_train, y_train)
    parallelPlot(df_all, color_column='Mean CV Accuracy', invert_column=cv_objs_max).to_html('all.html')
    df_non_dom = nondomSort(df_all, cv_objs, max_objs=cv_objs_max)
    parallelPlot(df_non_dom, color_column='Mean CV Accuracy', invert_column=cv_objs_max).to_html('non_dom.html')
    # Non-Dominated Set Test Performance
    df_non_dom_test = df_non_dom.apply(lambda row: getTestPerformance(X_train, X_test, y_train, y_test, row), axis=1)
    df_non_dom = df_non_dom.join(df_non_dom_test)
    # Check if Objective Performance is Preserved by Looking at Sorted Objective Values
    for i, j in zip(cv_objs, test_objs):
        print(df_non_dom[[i, j]].sort_values(i, ascending=False))
    return 0


if __name__ == '__main__':
    main()
