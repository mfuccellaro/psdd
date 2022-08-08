import pandas as pd
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd

def compute_rule(sample, feature, sign, value):
    """
    Helper function to process sklearn decision tree
    """
    try:
        value = float(value)
    except:
        pass
    if sign=='>':
        res = (sample[feature] > value)
    if sign=='>=':
        res = (sample[feature] >= value)
    if sign=='<=':
        res = (sample[feature] <= value)
    if sign=='<':
        res = (sample[feature] < value)
    if sign== '==':
        res = (sample[feature] == value)
    return res

def get_node(sample, rules):
    """
    Helper function to process sklearn decision tree
    """
    chain = '|---'
    i = 0
    while i<len(rules):
        rule = rules[i].replace('  ', ' ')
        if rule[0:len(chain)] == chain:
            if 'class' in rule:
                return i
            feature = rule.split(' ')[-3]
            sign =  rule.split(' ')[-2]
            value = rule.split(' ')[-1]
            if compute_rule(sample, feature, sign, value):
                i+=1
                chain = '|  '+chain
            else:
                i+=1
        else:
            i+=1
            

def PSDD_train(train, y_train, min_samples_split=20):
    """
    Train a decision tree, and compute training observations distribution.
    
    Parameters
    ----------
    train : Pandas DataFrame
        Training dataset used to train detector 
    y_train : Pandas Series
        Target Variable of training dataset
    min_samples_split : int
        Minimum number of observations per leaf
    
    Returns
    -------
    rules : str
        Built tree
    data_dict_train : dict
        Leaf information, contains leaf id with training samples
    """
    train_set = train.copy()
    y_train_set = y_train.copy()
    tree = DecisionTreeClassifier(min_samples_split=min_samples_split)
    tree.fit(train, y_train)
    # Export tree info
    s = export_text(tree, feature_names=list(train.columns), max_depth=150)
    rules = s.split('\n')[:-1]
    # get leaf for each dataset sample
    nodes = []
    for i in train_set.index:
        sample = train_set.loc[i]
        nodes.append(get_node(sample, rules))
    train_set['node'] = nodes
    
    train_set['class'] = y_train
    tmp = train_set.groupby(['node'])['class'].mean().reset_index()
    tmp = tmp[(tmp['class'] < min_impurity) | (tmp['class'] > 1-min_impurity)]
    del train_set['class']
    # Build KDTrees for each node
    data_dict_train = {}
    for i in list(tmp.node):
        train_ = train_set[train_set.node == i].copy()
        if len(train_) >= min_samples_split:
            del train_['node']
            data_dict_train[i] = train_
    return rules, data_dict_train


def PSDD_test(test, rules, data_dict_train, min_sample_number=20):
    """
    Compute the distribution of the inference set accross the trained tree.
    
    Parameters
    ----------
    test : Pandas DataFrame
        Inference dataset to detect drift upon
    rules : str
        Tree built un PSDD_train
    data_dict_train : dict
        Minimum number of observations per leaf
    min_sample_number : int
        Minimum number of observations for a leaf to be considered during analysis
    
    Returns
    -------
    data_dict_test : dict
        Leaf information, contains leaf id with inference samples
    """
    # Score Tree to get leaf for each test sample
    W2 = []
    data_dict_test = {}
    for i in test.index:
        sample = test.loc[i]
        W2.append(get_node(sample, rules))
    test['node'] = W2
    # Score KDTrees for each node
    for i in test.node.unique():
        if i in data_dict_train.keys():
            try:
                test_ = test[test.node == i].copy()
                if len(test_) >= min_sample_number:
                    del test_['node']
                    data_dict_test[i] = test_
            except:
                pass 
    return data_dict_test

def compute_stat(data_dict_train, data_dict_test, alpha, beta, test_type='mean'):
    """
    Compute stats to detect drift.
    
    Parameters
    ----------
    data_dict_train : dict
        Training dataset used to train detector 
    data_dict_test : dict
        Target Variable of training dataset
    alpha : float
        Alpha parameter, to reject or accept the null hypothesis
    beta : float
        Beta parameter, ratio of leaves to have a drift in order to be a drift
    Returns
    -------
    drift : bool
        True if a drift is detected
    """
    if test_type == 'mean':
        results_drift=[]
        for j in data_dict_test.keys():
            results_drift.append(
                stats.ttest_ind(
                    data_dict_test[j].mean(), data_dict_train[j].mean(), equal_var=False
                ).pvalue
            )
        results_drift = pd.DataFrame(results_drift)
        # print(results_drift)
        # ratio of leaves where null hypothesis is rejected
        if len(results_drift) > 0:
            val = float((results_drift < alpha).sum()/results_drift.size)
            drift = val >= beta
        else: 
            drift = False
        # drift flag if ratio > beta
    return drift