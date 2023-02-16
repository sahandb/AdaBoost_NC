# imports
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import scipy.stats as ss


def generateNoise(data, rc, rn):
    # number of sample
    nos = np.random.choice(data.shape[0], int(data.shape[0] * rn), replace=False)
    # number of feature
    nof = np.random.choice(data.shape[1], int(data.shape[1] * rc), replace=False)

    for itemNof in range(len(nof)):
        x = np.arange(min(data[nos, nof[itemNof]]), max(data[nos, nof[itemNof]]) + 1)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        for itemNos in range(len(nos)):
            data[nos[itemNos], nof[itemNof]] = np.random.choice(x, size=1, p=prob)
    return data


def normalSample(data):
    Y = data['Label']
    x = data.drop('Label', 1)
    return np.array(x), np.array(Y)


def randUndSampler(data):
    len_class_minority = len(data.loc[data['Label'] == 1])
    data_underSample_majority = data.loc[data['Label'] == -1].sample(n=len_class_minority, random_state=1)
    underSampled = pd.concat([data.loc[data['Label'] == 1], data_underSample_majority])
    Y = underSampled['Label']
    x = underSampled.drop('Label', 1)
    return np.array(x), np.array(Y)


def randOverSampler(data):
    len_class_majority = len(data.loc[data['Label'] == -1])
    data_overSample_minority = data.loc[data['Label'] == 1].sample(n=len_class_majority, random_state=1, replace=True)
    overSampled = pd.concat([data.loc[data['Label'] == -1], data_overSample_minority])
    Y = overSampled['Label']
    x = overSampled.drop('Label', 1)
    return np.array(x), np.array(Y)


# Returns error given the prediction and the input
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


# Returns accuracy of given the prediction and the input
def get_acc_rate(pred, Y):
    return sum(pred == Y) / float(len(Y))


# fix nan data(get dataframe and return new dataframe with no nan data)
def formatData(data):
    newDF = pd.DataFrame()  # creates a new dataframe that's empty
    for col_name in data.columns:
        newDF[col_name] = data[col_name].fillna(data[col_name].mode()[0])
        # check nulls
        # print("column:", col_name, ".Missing:", sum(data[col_name].isnull()))
    return newDF


# A function that takes classifier as input and performs classification
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)


# Adaboost algorithm implementation
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf, er_train, er_test):
    alpha_list = list()
    alphaT_list = list()
    y_predict_list_test = list()
    estimator_list_train = list()
    weight_list = list()
    trees = list()
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pt = np.ones(n_train)  # penalty term
    amb = np.zeros(n_train)
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    alpha_m = 0
    alpha_T = 0
    labels = np.unique(Y_train)
    strength = 1
    for i in range(M):
        # step 1
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        estimator_list_train.append(pred_train_i)
        pred_test_i = clf.predict(X_test)

        # step 2
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        right = [int(x) for x in (pred_train_i == Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x == 1 else -1 for x in miss]
        right2 = [x if x == 1 else -1 for x in right]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        rght_m = np.dot(w, right) / sum(w)

        # step 3
        # calculate the penalty value for every example
        if len(trees) == 0:
            p = 1 - np.abs(np.zeros(n_train))
        else:
            ht = np.zeros((n_train, len(labels)))
            for index, cls in enumerate(labels):
                ht[:, index] = np.sum(
                    [weight * (estimator.predict(X_train) == cls) for estimator, weight in zip(trees, alpha_list)],
                    axis=0)
            best = labels[np.argmax(ht, axis=1)]
            p = 1 - np.abs(np.mean(
                [(1 * (best == Y_train) - 1 * (tree.predict(X_train) == Y_train)) for tree in trees],
                axis=0))

        trees.append(clf)

        # # step 4: calculate Alpha by error and penalty
        correct = np.where(Y_train == pred_train_i)[0]
        incorrect = np.where(Y_train != pred_train_i)[0]
        alpha_T = 0.5 * (np.log((np.sum(np.multiply(w[correct], np.power(p[correct], strength)))) / (
            np.sum(np.multiply(w[incorrect], np.power(p[incorrect], strength))))))
        # Alpha
        # alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        alpha_m = 0.5 * np.log(float(rght_m + np.finfo(float).eps) / float(err_m + np.finfo(float).eps))
        alpha_list.append(alpha_T)
        # alphaT_list.append(alpha_T)

        # step 5: update data weights and obtain new weights by error and penalty
        # New weights
        # w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = (w * np.power(p, strength) * np.exp(-alpha_list[-1] * (pred_train_i == Y_train)))
        w /= np.sum(w)
        weight_list.append(w)
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_i])]
        y_predict_list_test.append(pred_test)

        pred_train1, pred_test1 = np.sign(pred_train), np.sign(pred_test)
        # Return error rate in train and test set
        er_train.append(get_error_rate(pred_train1, Y_train))
        er_test.append(get_error_rate(pred_test1, Y_test))

    trees_lisT = np.asarray(trees)
    alphaLisT = np.asarray(alpha_list)
    alphaTLisT = np.asarray(alphaT_list)
    estimator_list_traiN = np.asarray(estimator_list_train)

    # pred_yTest = list()
    # for item in x_test.values:
    #     accuracy = 0
    #     for li in range(estimator_list_traiN.shape[0]):
    #         accuracy += trees_lisT[li].predict(item.reshape(1, item.shape[0])) * alphaLisT[li]
    #     if accuracy >= 0:
    #         pred_yTest.append(1)
    #     else:
    #         pred_yTest.append(-1)

    htT = np.zeros((n_test, len(labels)))
    for indexTest, cLabel in enumerate(labels):
        htT[:, indexTest] = np.sum(
            [weightTest * (classifiersTest.predict(X_test) == cLabel) for classifiersTest, weightTest in
             zip(trees, alpha_list)],
            axis=0)
    y_predict_test_ = labels[np.argmax(htT, axis=1)]

    return er_train, er_test, y_predict_test_


def run(iter, settt, genNoise, rc, rn, preData,saveFile):
    # lists
    precision_list = list()
    recall_list = list()
    f1_list = list()
    auc_list = list()
    accuracy_list = list()
    g_mean_list = list()
    for runn in range(iter):
        # read data
        df = pd.read_csv('Covid-19.csv')
        noNan_data = formatData(df)
        # type of sampling
        if preData == 'normal':
            newX, newY = normalSample(noNan_data)
        elif preData == 'over':
            newX, newY = randOverSampler(noNan_data)
        elif preData == 'under':
            newX, newY = randUndSampler(noNan_data)

        # if else for noise generating
        if genNoise == 'none':
            # Split into training and test set
            x_train, x_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3)
            # Fit a simple decision tree first
            clf_tree = DecisionTreeClassifier(random_state=1)
            er_tree = generic_clf(y_train, x_train, y_test, x_test, clf_tree)

            # Fit Adaboost classifier using a decision tree stump
            er_train, er_test = [er_tree[0]], [er_tree[1]]
            op = adaboost_clf(y_train, x_train, y_test, x_test, settt, clf_tree, er_train, er_test)

            # minor class reports
            convert_ypred = 1 * (op[2] == 1)
            convert_y_2to1 = 1 * (y_test == 1)
            report = prfs(y_test, op[2], average='binary')
            fpr, tpr, thresh = roc_curve(y_test, op[2])
            report_auc = auc(fpr, tpr)
            report_accuracy = get_acc_rate(op[2], y_test) * 100
            gmean = np.sqrt(np.multiply(tpr, fpr)) * 100

            precision_list.append(report[0])
            recall_list.append(report[1])
            f1_list.append(report[2])
            auc_list.append(report_auc)
            accuracy_list.append(report_accuracy)
            g_mean_list.append(gmean[1])
            saveFile.write('\nPrecision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
                report[0] * 100,
                report[1] * 100,
                report[2] * 100,
                report_auc,
                report_accuracy,
                gmean[1]))
            print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
                report[0] * 100,
                report[1] * 100,
                report[2] * 100,
                report_auc,
                report_accuracy,
                gmean[1]))
        elif genNoise == 'train':
            # Split into training and test set
            x_train, x_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3)

            # inject noise
            x_trainNoisy = generateNoise(x_train, rc, rn)
            # x_testNoisy = generateNoise(x_test, rc, rn)

            # Fit a simple decision tree first
            clf_tree = DecisionTreeClassifier(random_state=1)
            er_tree = generic_clf(y_train, x_trainNoisy, y_test, x_test, clf_tree)

            # Fit Adaboost classifier using a decision tree stump
            er_train, er_test = [er_tree[0]], [er_tree[1]]
            op = adaboost_clf(y_train, x_trainNoisy, y_test, x_test, settt, clf_tree, er_train, er_test)

            # minor class reports
            convert_ypred = 1 * (op[2] == 1)
            convert_y_2to1 = 1 * (y_test == 1)
            report = prfs(y_test, op[2], average='binary')
            fpr, tpr, thresh = roc_curve(y_test, op[2])
            report_auc = auc(fpr, tpr)
            report_accuracy = get_acc_rate(op[2], y_test) * 100
            gmean = np.sqrt(np.multiply(tpr, fpr)) * 100

            precision_list.append(report[0])
            recall_list.append(report[1])
            f1_list.append(report[2])
            auc_list.append(report_auc)
            accuracy_list.append(report_accuracy)
            g_mean_list.append(gmean[1])
            saveFile.write('\nPrecision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
                report[0] * 100,
                report[1] * 100,
                report[2] * 100,
                report_auc,
                report_accuracy,
                gmean[1]))
            print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
                report[0] * 100,
                report[1] * 100,
                report[2] * 100,
                report_auc,
                report_accuracy,
                gmean[1]))
        elif genNoise == 'test':
            # Split into training and test set
            x_train, x_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3)

            # inject noise
            # x_trainNoisy = generateNoise(x_train, rc, rn)
            x_testNoisy = generateNoise(x_test, rc, rn)

            # Fit a simple decision tree first
            clf_tree = DecisionTreeClassifier(random_state=1)
            er_tree = generic_clf(y_train, x_train, y_test, x_testNoisy, clf_tree)

            # Fit Adaboost classifier using a decision tree stump
            er_train, er_test = [er_tree[0]], [er_tree[1]]
            op = adaboost_clf(y_train, x_train, y_test, x_testNoisy, settt, clf_tree, er_train, er_test)

            # minor class reports
            convert_ypred = 1 * (op[2] == 1)
            convert_y_2to1 = 1 * (y_test == 1)
            report = prfs(y_test, op[2], average='binary')
            fpr, tpr, thresh = roc_curve(y_test, op[2])
            report_auc = auc(fpr, tpr)
            report_accuracy = get_acc_rate(op[2], y_test) * 100
            gmean = np.sqrt(np.multiply(tpr, fpr)) * 100

            precision_list.append(report[0])
            recall_list.append(report[1])
            f1_list.append(report[2])
            auc_list.append(report_auc)
            accuracy_list.append(report_accuracy)
            g_mean_list.append(gmean[1])
            saveFile.write('\nPrecision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
                report[0] * 100,
                report[1] * 100,
                report[2] * 100,
                report_auc,
                report_accuracy,
                gmean[1]))
            print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
                report[0] * 100,
                report[1] * 100,
                report[2] * 100,
                report_auc,
                report_accuracy,
                gmean[1]))

    saveFile.write("\n\n\n============================= mean set: =============================")
    saveFile.write('\nPrecision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5} \n'.format(
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(f1_list),
        np.mean(auc_list),
        np.mean(accuracy_list),
        np.mean(g_mean_list)))
    saveFile.write("\n============================= std set: =============================")
    saveFile.write('\nPrecision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5} \n'.format(
        np.std(precision_list),
        np.std(recall_list),
        np.std(f1_list),
        np.std(auc_list),
        np.std(accuracy_list),
        np.std(g_mean_list)))

    print("\n\n============================= mean set: =============================")
    print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5} \n'.format(
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(f1_list),
        np.mean(auc_list),
        np.mean(accuracy_list),
        np.mean(g_mean_list)))
    print("============================= std set: =============================")
    print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5} \n'.format(
        np.std(precision_list),
        np.std(recall_list),
        np.std(f1_list),
        np.std(auc_list),
        np.std(accuracy_list),
        np.std(g_mean_list)))


def main():
    sets = [11, 21, 31, 41, 51]
    runner = 10

    # for noise
    rc = [0.1, 0.3, 0.5]
    rn = [0.1, 0.3, 0.5]

    # type Of sampler
    # 'normal' , 'under' , 'over'
    sampler = 'over'
    file = open('report.txt','a')

    print("=======================================normal data=======================================\n")
    file.write("\n=======================================normal data=======================================\n")
    for sett in sets:
        print(
            "\n\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(sett))
        file.write(
            "\n\n\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(sett))
        run(runner, sett, genNoise='none', rc=0, rn=0, preData=sampler,saveFile=file)
    print("\n\n=======================================now start noisy data=======================================\n\n")
    file.write("\n\n\n=======================================now start noisy data=======================================\n\n")
    for sett in sets:
        for rcc in rc:
            for rnn in rn:
                print('\n\ngenerate noise with rc = {0} and rn = {1} for only noisy train data'.format(rcc, rnn))
                file.write('\n\n\ngenerate noise with rc = {0} and rn = {1} for only noisy train data'.format(rcc, rnn))
                print(
                    "\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(
                        sett))
                file.write(
                    "\n\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(
                        sett))
                run(runner, sett, genNoise='train', rc=rcc, rn=rnn, preData=sampler,saveFile=file)
    for sett in sets:
        for rcc in rc:
            for rnn in rn:
                print('\n\ngenerate noise with rc = {0} and rn = {1} for only noisy test data'.format(rcc, rnn))
                file.write('\n\n\ngenerate noise with rc = {0} and rn = {1} for only noisy test data'.format(rcc, rnn))
                print(
                    "\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(
                        sett))
                file.write(
                    "\n\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(
                        sett))
                run(runner, sett, genNoise='test', rc=rcc, rn=rnn, preData=sampler,saveFile=file)
    file.close()


# Driver
if __name__ == '__main__':
    main()
