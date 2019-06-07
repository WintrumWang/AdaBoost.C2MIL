# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from sklearn import metrics
MIN_SAMPLE=sys.argv[1]
EPSILON=sys.argv[2]
NUM_ITER=sys.argv[3]

raw_data = pd.read_csv("./data.csv", header=None)
raw_data = np.array(raw_data)
sam_label = raw_data[:,[0,2]]
raw_data = np.delete(raw_data,0,axis=1)
test_group_num = int(len(np.unique(raw_data[:,1]))/5)
test_group = random.sample(list(np.unique(raw_data[:,1])),test_group_num)
raw_data_test = raw_data[np.where(np.isin(raw_data[:,1],test_group))]
sam_label_test = sam_label[np.where(np.isin(sam_label[:,1],test_group))]
raw_data_train = raw_data[np.where(np.isin(raw_data[:,1],test_group) == False)]
sample_num = raw_data_train.shape[0]
small_num = 1/sample_num

def calSumWP(dataSet):
    return np.sum(dataSet[np.where(dataSet[:, -2] == 1), -1])

def calSumWN(dataSet):
    return np.sum(dataSet[np.where(dataSet[:, -2] != 1), -1])

def calImpurity(dataSet):
    w_p = calSumWP(dataSet)
    w_n = calSumWN(dataSet)
    return (w_p * w_n) ** 0.5

def calHypo(dataSet):
    w_p = calSumWP(dataSet)
    w_n = calSumWN(dataSet)
    return 0.5 * np.log((w_p + small_num) / (w_n + small_num))

def splitDataset(dataSet, feat, value):
    dataSet1 = dataSet[dataSet[:, feat] <= value]
    dataSet2 = dataSet[dataSet[:, feat] > value]
    return dataSet1, dataSet2

def chooseCriterion(dataSet, min_sample=MIN_SAMPLE, epsilon=EPSILON):
    feat_num = dataSet.shape[1] - 2
    sImpurity = calImpurity(dataSet)
    bestColumn = 0
    bestValue = 0
    minImpurity = np.inf
    if len(np.unique(dataSet[:, -2])) == 1:
        return None, calHypo(dataSet)
    for feat in range(feat_num):
        if len(np.unique(dataSet[:,feat])) <= 20:
            for row in range(dataSet.shape[0]):
                dataSet1, dataSet2 = splitDataset(dataSet, feat, dataSet[row, feat])
                if len(dataSet1) < min_sample or len(dataSet2) < min_sample:
                    continue
                nowImpurity = calImpurity(dataSet1) + calImpurity(dataSet2)
                if nowImpurity < minImpurity:
                    minImpurity = nowImpurity
                    bestColumn = feat
                    bestValue = dataSet[row, feat]
        else:
            candidate = np.linspace(np.min(dataSet[:,feat]),np.max(dataSet[:,feat]),20)
            for candi in candidate:
                dataSet1, dataSet2 = splitDataset(dataSet, feat, candi)
                if len(dataSet1) < min_sample or len(dataSet2) < min_sample:
                    continue
                nowImpurity = calImpurity(dataSet1) + calImpurity(dataSet2)
                if nowImpurity < minImpurity:
                    minImpurity = nowImpurity
                    bestColumn = feat
                    bestValue = candi
    if (sImpurity - minImpurity) < epsilon:
        return None, calHypo(dataSet)
    dataSet1, dataSet2 = splitDataset(dataSet, bestColumn, bestValue)
    if len(dataSet1) < min_sample or len(dataSet2) < min_sample:
        return None, calHypo(dataSet)
    return bestColumn, bestValue

def buildTree(dataSet):
    bestColumn, bestValue = chooseCriterion(dataSet)
    if bestColumn is None:
        return bestValue
    tree = {}
    tree['spCol'] = bestColumn
    tree['spVal'] = bestValue
    lSet, rSet = splitDataset(dataSet, bestColumn, bestValue)
    tree['left'] = calHypo(lSet)
    tree['right'] = calHypo(rSet)
    return tree

def predictHypo(dataSet, tree):
    if type(tree) is not dict:
        return tree
    if dataSet[tree['spCol']] <= tree['spVal']:
        return tree['left']
    else:
        return tree['right']

def outputHypo(dataSet, each_tree):
    data_num = dataSet.shape[0]
    hypo = np.zeros((data_num,), dtype=np.float32)
    for i in range(data_num):
        hypo[i] = predictHypo(dataSet[i], each_tree)
    return hypo

def sigmoid(x):
    return 1/(1+np.exp(-x))

def integrate(df):
    return np.sum(np.multiply(np.sign(df['sample_hypo']), sigmoid(df['sample_hypo'])))/np.sum(sigmoid(df['sample_hypo']))

def adaboostTrainer(raw_data, num_iter):
    positive_index = np.where(raw_data[:, 0] == 1)[0]
    negative_index = np.where(raw_data[:, 0] != 1)[0]
    raw_data[negative_index, 1] = np.linspace(np.max(raw_data[:, 1]), np.max(raw_data[:, 1])+negative_index.shape[0], negative_index.shape[0], dtype=np.int64)
    data = raw_data[:, 2:]
    label = raw_data[:, 0]
    group = raw_data[:, 1]
    base_learnerArray = []
    m, n = np.shape(data)
    group_num = np.unique(group).shape[0]
    group_joint_label = pd.DataFrame({'group': group, 'label': label})
    group_label = np.array(group_joint_label.groupby(['group']).max()).reshape(group_num,)
    group_weight = np.hstack((np.unique(group).reshape(group_num, 1), np.array(np.ones((group_num, 1)), dtype=np.float32)/group_num))
    sample_predictionArray = np.zeros((m,), dtype=np.float32)
    group_predictionArray = np.zeros((group_num,), dtype=np.float32)
    while num_iter:
        sample_weight = np.zeros((m, 1), dtype=np.float32)
        for row in range(m):
            sample_weight[row, 0] = group_weight[np.where(group_weight[:, 0] == group[row])[0][0], 1] / np.sum(group == group[row])
        tran_data = np.hstack((data, label.reshape(m, 1), sample_weight))
        base_tree = buildTree(tran_data)
        sample_hypo = outputHypo(tran_data, base_tree)
        df = pd.DataFrame({'group': group, 'sample_hypo': sample_hypo})
        group_hypo = np.array(df.groupby(['group']).apply(lambda df: integrate(df)))
        loss = np.multiply(np.sign(group_hypo), group_label)
        error = np.sum(group_weight[np.where(loss == -1), 1])
        alpha = np.log((1 - error) / error) / 2
        base_learnerArray.append({'tree': base_tree, 'alpha': alpha})
        sample_predictionArray += sample_hypo*alpha
        group_predictionArray += group_hypo*alpha
        print(num_iter)
        print(base_tree)
        num_iter -= 1
        expon = np.multiply(-alpha*np.sign(group_hypo), group_label)
        group_weight[:, 1] = np.multiply(group_weight[:, 1], np.exp(expon))
        group_weight[:, 1] = group_weight[:, 1] / np.sum(group_weight[:, 1])
    sample_finalPrediction = np.sign(sample_predictionArray)
    group_finalPrediction = np.sign(group_predictionArray)
    return base_learnerArray

def adaboostPredictor(data_predict, base_learnerArray):
    group_predict = data_predict[:, -1]
    pre_sample_num = data_predict.shape[0]
    pre_group_num = np.unique(group_predict).shape[0]
    sample_hypoPredict = np.zeros((pre_sample_num,), dtype=np.float32)
    group_hypoPredict = np.zeros((pre_group_num,), dtype=np.float32)
    alpha_sum = 0
    for base_learner in base_learnerArray:
        sample_hypoBase = outputHypo(data_predict, base_learner['tree'])
        df = pd.DataFrame({'group': group_predict, 'sample_hypo': sample_hypoBase})
        group_hypoBase = np.array(df.groupby(['group']).apply(lambda df: integrate(df)))
        sample_hypoPredict += base_learner["alpha"] * sample_hypoBase
        group_hypoPredict += base_learner["alpha"] * group_hypoBase
        alpha_sum += base_learner["alpha"]
    sample_finalPredict = np.sign(sample_hypoPredict)
    group_finalPredict = np.sign(group_hypoPredict)
    sample_hypoProb = sigmoid(sample_hypoPredict / alpha_sum)
    group_hypoProb = sigmoid(group_hypoPredict / alpha_sum)
    return sample_finalPredict, group_finalPredict, sample_hypoProb, group_hypoProb

def main():
    tree = adaboostTrainer(raw_data_train, num_iter=NUM_ITER)
    data_predict = np.hstack((raw_data_test[:,2:], raw_data_test[:,1].reshape(raw_data_test.shape[0],1)))
    sam, gro, samprob, groprob = adaboostPredictor(data_predict, tree)
    sam_truth = sam_label_test[:, 0]
    gro_truth = np.array(pd.DataFrame(raw_data_test[:, 0:2]).groupby([1]).mean()).T[0]
    sam_acc = np.sum(sam_truth == sam)/raw_data_test.shape[0]
    gro_acc = np.sum(gro_truth == gro)/test_group_num
    sam_auc = metrics.roc_auc_score(sam_truth, samprob)
    gro_auc = metrics.roc_auc_score(gro_truth, groprob)
    print(sam_acc, gro_acc, sam_auc, gro_auc)

if __name__ == '__main__':
    main()
