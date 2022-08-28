from load_data import load_data
from pre_process import split_dataset, to_bag_of_words, to_id_list, to_word_vector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import random
import os
from sklearn.model_selection import ParameterGrid
import joblib


# 寻找随机森林最好的超参数
def find_best_hyper_parameter(all_data):
    # dataset = split_dataset(to_bag_of_words(all_data[:10000])) # 此处只用前一万条数据搜索参数
    # dataset = split_dataset(to_word_vector(all_data[:10000]))
    dataset = split_dataset(to_id_list(all_data[:10000]))
    train_set, val_set, test_set = dataset
    # 设定超参数搜索范围
    param_grid = {'n_estimators': [10, 100], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 10, None],
                  'min_samples_leaf': [1, 3, 5], 'max_features': ['sqrt', 'log2', 0.5, 0.8]}
    grid = ParameterGrid(param_grid)
    max_acc = 0.
    for params in grid:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'],
                                     max_depth=params['max_depth'], min_samples_leaf=params['min_samples_leaf'],
                                     max_features=params['max_features'], random_state=2021, n_jobs=-1)
        clf.fit(train_set[0], train_set[1])
        acc = accuracy_score(val_set[1], clf.predict(val_set[0]))
        # 输出此组参数的预测准确率
        print('Accuracy: {:.3f}, Parameters: {:s}'.format(acc, str(params)))
        # 找到当前最好的超参数
        if acc > max_acc:
            max_acc = acc
            best_params = params
            print('Maximum Accuracy!')
    # 搜索完成，输出最好的超参
    print('Maximum Accuracy: {:.3f}, Best Parameters: {:s}'.format(max_acc, str(best_params)))


# 使用搜索好的超超参数在全部数据上训练模型
def train(all_data):
    dataset = split_dataset(to_bag_of_words(all_data))
    train_set, val_set, test_set = dataset
    print(train_set[0].shape, train_set[1].shape)
    print(val_set[0].shape, val_set[1].shape)
    print(test_set[0].shape, test_set[1].shape)

    clf = RandomForestClassifier(n_estimators=10, criterion='entropy',
                                 max_depth=None, min_samples_leaf=1,
                                 max_features='log2', random_state=2021, n_jobs=-1)
    clf.fit(train_set[0], train_set[1])
    # 保存模型到本地
    joblib.dump(clf, 'random_forest.model')


if __name__ == '__main__':
    if os.path.exists('vocab.json'):
        os.remove('vocab.json')
    if os.path.exists('bow_x.npy'):
        os.remove('bow_x.npy')
    if os.path.exists('bow_y.npy'):
        os.remove('bow_y.npy')
    if os.path.exists('wordvec_x.npy'):
        os.remove('wordvec_x.npy')
    if os.path.exists('wordvec_y.npy'):
        os.remove('wordvec_y.npy')
    if os.path.exists('idlist_x.npy'):
        os.remove('idlist_x.npy')
    if os.path.exists('idlist_y.npy'):
        os.remove('idlist_y.npy')

    seed = 2021
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    all_data = load_data()
    random.shuffle(all_data)

    find_best_hyper_parameter(all_data)
    # train(all_data)
