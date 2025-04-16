import numpy as np
import pandas
import json
import os

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    else:
        return obj

def get_metric(cv_criteria, average='macro'):
    if cv_criteria == 'roc_auc':
        return metrics.make_scorer(metrics.roc_auc_score, average=average, greater_is_better=True, multi_class='ovr', needs_proba=True)
    elif cv_criteria == 'recall':
        return metrics.make_scorer(metrics.recall_score, average=average, greater_is_better=True)
    elif cv_criteria == 'precision':
        return metrics.make_scorer(metrics.precision_score, average=average, greater_is_better=True)
    elif cv_criteria == 'f1':
        return metrics.make_scorer(metrics.f1_score, average=average, greater_is_better=True)
    else:
        raise ValueError("Invalid cv_criteria")

def load_incivility_dataset(y_label):
    data_path = Path(r"./data")

    df = pandas.read_csv(Path(data_path / 'reference_dataset_fg.csv'),  na_values=[""], keep_default_na=False)
    encoder = {k:v for v, k in enumerate(df[y_label].unique())}
    print("encoder:", encoder)
    df[y_label] = df[y_label].replace(encoder)
    df[y_label] = df[y_label].astype(int)
    df = df.dropna(subset=[y_label])
    return df

def compute_metrics_scores_binary(y_test, y_pred): 
    scores = {"accuracy_score":metrics.accuracy_score(y_test, y_pred),
              "roc_auc_score": metrics.roc_auc_score(y_test, y_pred),
              "f1_score":metrics.f1_score(y_test, y_pred),
              "precision_score":metrics.precision_score(y_test, y_pred),
              "recall_score":metrics.recall_score(y_test, y_pred),
              "matthews_corrcoef":metrics.matthews_corrcoef(y_test, y_pred),
              "brier_score_loss":metrics.brier_score_loss(y_test, y_pred),
              "confusion_matrix":metrics.confusion_matrix(y_test, y_pred),
              "classification_report":metrics.classification_report(y_test, y_pred)}
    return scores

def compute_and_save_classification_report(y_test, y_pred):
    report_df = metrics.classification_report(y_test, y_pred, output_dict=True)
    report_df = pandas.DataFrame(report_df).transpose()
    return report_df

def compute_metrics_scores_multiclass(y_test, y_pred, y_proba) :
    scores = {"accuracy_score":metrics.accuracy_score(y_test, y_pred),
              "roc_auc_score": metrics.roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'),
              "f1_score":metrics.f1_score(y_test, y_pred, average='weighted'),
              "precision_score":metrics.precision_score(y_test, y_pred, average='weighted'),
              "recall_score":metrics.recall_score(y_test, y_pred, average='weighted'),
              "matthews_corrcoef":metrics.matthews_corrcoef(y_test, y_pred),
              "confusion_matrix":metrics.confusion_matrix(y_test, y_pred),
              "classification_report":metrics.classification_report(y_test, y_pred)}
    return scores

def run_experiment(dataset, x_atributes, data_balance, y_label, models, grid_params_list, word_embedding, cv_criteria):
    dataset.dropna(subset = [x_atributes], inplace=True)
    X = dataset[x_atributes]
    y = dataset[y_label]

    skf = StratifiedKFold(n_splits=10)
    folds = []
    results_path = Path("results")
    resulst_ml_models_path = results_path / "ml_models"
    resulst_ml_models_path_folds = resulst_ml_models_path / "folds"
    resulst_ml_models_path_folds.mkdir(parents=True, exist_ok=True)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i+1}:")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if word_embedding=='BoW':
            vectorizer = CountVectorizer()
            vectorizer.fit(X_train.values)            
            bag_of_word_data = vectorizer.transform(X_train.values)            
            X_train = pandas.DataFrame(bag_of_word_data.toarray(),columns=vectorizer.get_feature_names_out())
            if data_balance=='SMOTE':
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)

            bag_of_word_data = vectorizer.transform(X_test.values)
            X_test = pandas.DataFrame(bag_of_word_data.toarray(),columns=vectorizer.get_feature_names_out())
        elif word_embedding=='TF-IDF':
            vectorizer = TfidfVectorizer()
            vectorizer.fit(X_train.values)
            tfidf_data = vectorizer.transform(X_train.values)
            X_train = pandas.DataFrame(tfidf_data.toarray(),columns=vectorizer.get_feature_names_out())
            if data_balance=='SMOTE':
                smote = SMOTE()
                X_train, y_train = smote.fit_resample(X_train, y_train)

            tfidf_data = vectorizer.transform(X_test.values)
            X_test = pandas.DataFrame(tfidf_data.toarray(),columns=vectorizer.get_feature_names_out())
        
        results = {}
        fold_path = resulst_ml_models_path_folds / f"fold-{i+1}"
        os.makedirs(fold_path, exist_ok=True)

        for model in models:
            print(f"{word_embedding}: {model}")
            print("All features...")
            curr_model = models[model]                       
            curr_model.fit(X_train, y_train)
            y_pred = curr_model.predict(X_test)

            results['fold'] = i+1   
            if y_label == 'tbdf':
                try:
                    # tenta pegar as probabilidades
                    y_proba = curr_model.predict_proba(X_test)
                except AttributeError:
                    # fallback para decision_function
                    try:
                        y_proba = curr_model.decision_function(X_test)
                    except AttributeError:
                        print(f"Modelo {model} não suporta predict_proba nem decision_function.")
                        y_proba = None

                if y_proba is not None:
                    results[model] = compute_metrics_scores_multiclass(y_test, y_pred, y_proba)
                else:
                    results[model] = compute_metrics_scores_multiclass(y_test, y_pred, None)
                
                word_embedding_path = fold_path / word_embedding
                os.makedirs(word_embedding_path, exist_ok=True)
                classification_report_path = word_embedding_path / f"classification_report-{word_embedding}-{cv_criteria}-{data_balance}-{model}-fold-{i+1}.csv"
                classification_report_df = compute_and_save_classification_report(y_test, y_pred)
                classification_report_df.to_csv(classification_report_path, index=True)
            else:
                results[model] = compute_metrics_scores_binary(y_test, y_pred)   

                word_embedding_path = fold_path / word_embedding
                os.makedirs(word_embedding_path, exist_ok=True)
                classification_report_path = word_embedding_path / f"classification_report-{word_embedding}-{cv_criteria}-{data_balance}-{model}-fold-{i+1}.csv"
                classification_report_df = compute_and_save_classification_report(y_test, y_pred)
                classification_report_df.to_csv(classification_report_path, index=True)


        folds.append(results)
    print(folds)

    os.makedirs(results_path, exist_ok=True)
    with open(resulst_ml_models_path /  f"results-{word_embedding}-{cv_criteria}-{data_balance}-{'-'.join(models)}.json", "w", encoding="utf-8") as outfile:
        json.dump(convert_ndarray(folds), outfile, indent=4, ensure_ascii=False)

    return folds

def do_benchmark(grid_search=False, data_balance='OD', cv_criteria='roc_auc', selected_models=['SVC', 'MNB', 'LRC', 'RFC']):    
    x_atributes = 'comment_body'
    y_label = 'tbdf'
    dataset =  load_incivility_dataset(y_label)
    
    train_models = {"MNB":MultinomialNB(), 
              "LRC":LogisticRegression(max_iter=10**7),              
              "RFC":RandomForestClassifier(),
              "ADA": AdaBoostClassifier()}

    models = {i:train_models[i] for i in train_models if i in selected_models}
              
    if grid_search:            
        grid_params_list = {
                            "MNB":{'alpha': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5, 10.0, 15, 20, 25, 30, 35, 40]},
                            
                            "LRC":{"C":[0.001,0.005,0.01,0.05, 0.1, 0.5, 1, 5, 10], 
                                "penalty":["l1","l2"],                            
                                "max_iter":[10**7],
                                "fit_intercept":[True],
                                "solver":["liblinear"]},                                                    

                            "RFC":{"n_estimators":[5,30,50,75, 100, 150, 200],
                                "max_depth": [4,6,7]},             
            
                            "SVC":{"C":[0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                                "max_iter":[10**7]},                             
                             
                            }
    else:
        grid_params_list = {"MNB":{},"LRC":{"max_iter":[10**7]}, "RFC":{}}
    
    fold_results = run_experiment(dataset, x_atributes, data_balance, y_label, models, grid_params_list, 'BoW', cv_criteria)
    print("#### BoW ####")
    print(fold_results)
    fold_results = run_experiment(dataset, x_atributes, data_balance, y_label, models, grid_params_list, 'TF-IDF', cv_criteria)
    print("#### TF-IDF ####")
    print(fold_results)

if __name__ == '__main__':
    '''
        params:
            grid_search
            data_balance
            comments
    
    '''
    #setup the training params
    grid_search = False
    data_balance= 'OD'
    comments = True
    CV = 10
       
    do_benchmark(grid_search, data_balance, 'recall', ["ADA"])
    