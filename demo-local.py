import numpy as np
import pandas as pd
import shap
import os
import warnings
from utilities.timewiseCV import time_wise_CV
from utilities.AutoSpearman import AutoSpearman
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from utilities.tuneParameters import tune
import lime
from lime import lime_tabular

sampling_methods = {
    "none": None,
    'rus': RandomUnderSampler(random_state=0),
    'rom': RandomOverSampler(random_state=0),
    'smo': SMOTE(random_state=0),
}

def preprocessing(data):
    #preserve label and effort
    data_label = data['bug'].to_frame()
    data_effort = (data['la'] + data['ld']).to_frame()
    data_effort.columns = ['effort']
    data_time = data['commitTime']
    #remove label
    all_cols = data.columns
    for col in all_cols:
        if col in ['bug', 'commitTime', 'commitdate']:
            data = data.drop(col, axis=1)
    # Remove Correlation and Redundancy(AotuSpearman)
    data_feature = AutoSpearman(data, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5)
    ## Remove feature interaction(CFS)
    #idx = cfs(data_feature, data_label)
    num_feature=data_feature.shape[1]
    data = pd.concat((data_time, data_feature, data_effort, data_label), axis=1)
    # log transformation
    cols_to_normalize = data.columns.difference(['commitTime','fix','effort','bug'])
    data[cols_to_normalize] = np.log(data[cols_to_normalize] + 1)
    return data, num_feature

def time_wise_fold_divided(fold,train_folds,test_folds,gap_folds):
    #train
    train_label = train_folds[fold]['bug']
    train_data = train_folds[fold].drop(['effort','bug'], axis=1)
    #test
    LOC = test_folds[fold]['effort']
    test_label = test_folds[fold]['bug']
    test_data = test_folds[fold].drop(['effort','bug'], axis=1)
    #gap
    gap_label = gap_folds[fold]['bug']
    gap_data = gap_folds[fold].drop(['effort', 'bug'], axis=1)
    return train_data,train_label,test_data,test_label,gap_data,gap_label,LOC

def lime_explain_instance(explainer_lime, sample, model):
    lime_explain = explainer_lime.explain_instance(sample, model.predict_proba)
    score_lime = np.array(lime_explain.local_exp[1])
    score_lime = (score_lime[score_lime[:, 0].argsort()])[:, 1]
    # Add observation index and LIME values to DataFrame
    return score_lime

def save_shap_results_to_csv(results, project_name,feature_names,model_name):
    for key, value in results.items():
        df = pd.DataFrame(value)
        # Drop rows with NaN values
        df = df.dropna()
        # Reset the index
        df = df.reset_index(drop=True)
        df.columns = feature_names
        outpath = f'./output/local/shap_scores/{project_name}-{model_name}-{key}.csv'
        df.to_csv(outpath, index=True, header=True)

def save_lime_results_to_csv(results, project_name,feature_names,model_name):
    for key, value in results.items():
        df = pd.DataFrame(value)
        # Drop rows with NaN values
        df = df.dropna()
        # Reset the index
        df = df.reset_index(drop=True)
        df.columns = feature_names
        outpath = f'./output/local/lime_scores/{project_name}-{model_name}-{key}.csv'
        df.to_csv(outpath, index=True, header=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    save_path = r'./result-importance/'
    #project_names = sorted(os.listdir('./dataset/'))
    project_names = sorted(os.listdir('./dataset/'), reverse=True)  # reverse project_names
    path = os.path.abspath('./dataset/')
    pro_num = len(project_names)
    column_name = ['commitTime', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp',
                   'rexp', 'sexp', 'bug']
    model_names =['logistic_regression','naive_bayes', 'support_vector_machine', 'gradient_boosting', 'decision_tree', 'knn', 'random_forest', 'logistic_regression']
    for model_name in model_names:
        for i in range(0, pro_num):
            # read data
            project_name = project_names[i]
            file = os.path.join(path, project_name)
            data = pd.read_csv(file)
            project_name = project_name[:-4]
            data = data[column_name]
            #data preprocessing
            data,num_feature= preprocessing(data)
            #time wise
            gap = 2
            scores_lime = {key: np.zeros(shape=(0, num_feature))  for key in sampling_methods.keys()}
            scores_shap = {key: np.zeros(shape=(0, num_feature)) for key in sampling_methods.keys()}

            train_folds, test_folds, gap_folds = time_wise_CV(data, gap)
            for fold in range(len(train_folds)):
                train_data, train_label, test_data, test_label, gap_data, gap_label, LOC = time_wise_fold_divided(fold,train_folds,test_folds,gap_folds)
                # ensure train data: the number of defect > non defect, only one class
                if len(np.unique(train_label)) < 2 or list(train_label).count(1) < 6 or list(train_label).count(1) > list(train_label).count(0):
                    continue
                for method, sampler in sampling_methods.items():
                    if sampler is None:
                        n_X, n_y = train_data, train_label
                    else:
                        n_X, n_y = sampler.fit_resample(train_data, train_label)
                    # ensure test data is not single class
                    if list(n_y).count(1) < 2 or list(n_y).count(0) < 2:
                        break
                    #train model(tune para)
                    model = tune(n_X, n_y,gap_data, gap_label,model_name)
                    # shap value predicted as defect proneness
                    feature_names = test_data.columns

                    # shap_explain
                    if (model_name in ['naive_bayes','knn']):
                        explainer_shap = shap.Explainer(model.predict, test_data)
                    elif (model_name in ['random_forest','gradient_boosting','decision_tree']):
                        explainer_shap = shap.TreeExplainer(model, check_additivity=False)
                    elif (model_name in ['logistic_regression']):
                        explainer_shap = shap.Explainer(model, test_data)
                    else:
                        explainer_shap = shap.Explainer(model, check_additivity=False)#(model_name in ['gradient_boosting','decision_tree']):


                    ##LIME
                    # lime_explainer
                    explainer_lime = lime.lime_tabular.LimeTabularExplainer(n_X.values,
                                                                            feature_names=n_X.columns,
                                                                            class_names=['defective'],
                                                                            mode='classification',
                                                                            feature_selection='none',
                                                                            verbose=True,
                                                                            random_state=42)
                    # test sample explain
                    filtered_data = test_data[test_label != 0]
                    # for sample in filtered_data:
                    for i in range(filtered_data.shape[0]):
                        sample = filtered_data.iloc[i]

                        ##shap explain
                        # calculate local shap values
                        if (model_name in ['random_forest', 'decision_tree']):
                            shap_values = explainer_shap.shap_values(sample)[0]
                        elif (model_name in ['naive_bayes', 'support_vector_machine', 'knn']):
                            shap_sample = sample.to_numpy()
                            shap_sample = shap_sample.reshape(1, -1)
                            shap_values = explainer_shap(shap_sample).values
                        else:
                            shap_values = explainer_shap.shap_values(sample)
                        # Add observation index and SHAP values
                        scores_shap[method] = np.vstack((scores_shap[method], shap_values))

                        # lime instance explain
                        score_lime = lime_explain_instance(explainer_lime, sample, model)
                        # Add observation index and LIME values to DataFrame
                        # Add observation index and LIME values to DataFrame
                        scores_lime[method] = np.vstack((scores_lime[method], score_lime))


            save_shap_results_to_csv(scores_shap, project_name,feature_names,model_name)
            save_lime_results_to_csv(scores_lime, project_name, feature_names, model_name)
            print(f"{project_name} is okay~")
        print(f"{model_name} running is okay~")