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


# def randUndSampler(data):
#     Y = data['Label']
#     x = data.drop('Label', 1)
#     underSampler = RandomUnderSampler(random_state=0)
#     fitted = underSampler.fit_resample(x, Y)
#     return fitted

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


def randUndSampler(data):
    len_class_minority = len(data.loc[data['Label'] == 1])
    data_underSample_majority = data.loc[data['Label'] == -1].sample(n=len_class_minority, random_state=1)
    underSampled = pd.concat([data.loc[data['Label'] == 1], data_underSample_majority])
    Y = underSampled['Label']
    x = underSampled.drop('Label', 1)
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
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    alpha_m = 0
    alpha_list = list()
    y_predict_list_train = list()
    y_predict_list_test = list()
    estimator_error_list = list()
    estimator_list_train = list()
    estimator_list_test = list()
    weight_list = list()
    trees = list()

    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight=w)
        trees.append(clf)
        pred_train_i = clf.predict(X_train)
        estimator_list_train.append(pred_train_i)
        pred_test_i = clf.predict(X_test)
        estimator_list_test.append(pred_test_i)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights for update weights
        miss2 = [x if x == 1 else -1 for x in miss]
        # Estimate Error
        err_m = np.dot(w, miss) / sum(w)
        estimator_error_list.append(err_m)
        # Alpha (estimate weight)
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        alpha_list.append(alpha_m)
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        weight_list.append(w)

        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_i])]

        y_predict_list_train.append(pred_train)
        y_predict_list_test.append(pred_test)

        pred_train1, pred_test1 = np.sign(pred_train), np.sign(pred_test)
        # Return error rate in train and test set
        er_train.append(get_acc_rate(pred_train1, Y_train))
        er_test.append(get_acc_rate(pred_test1, Y_test))

    return er_train, er_test, \
           y_predict_list_train, \
           y_predict_list_test, \
           estimator_list_train, \
           estimator_list_test, \
           estimator_error_list, \
           alpha_list, \
           weight_list, \
           trees


# # Function to plot error vs number of iterations
# def plot_error_rate(er_train, er_test):
#     plt.plot(er_train, label="Training")
#     plt.plot(er_test, label="Test")
#     plt.xlabel('Number of iterations')
#     plt.legend()
#     plt.title('Error rate vs number of iterations')
#     plt.grid()
#     plt.show()

def adaboostUnderSampling(path, runn,rc,rn,genNoise):
    y_pred_list_train = list()
    y_pred_list_test = list()
    estimator_list_train = list()
    estimator_list_test = list()
    alphaList = list()
    weight_list = list()
    estimator_error_list = list()
    er_tree_list_train = list()
    er_tree_list_test = list()
    trees_list = list()

    precision_list = list()
    recall_list = list()
    f1_list = list()
    auc_list = list()
    accuracy_list = list()
    g_mean_list = list()

    # read data
    df = pd.read_csv(path)
    noNan_data = formatData(df)
    newX, newY = randUndSampler(noNan_data)

    # Split into training and test set
    x_train, x_test, y_train, y_test = train_test_split(newX, newY, test_size=0.2)
    # inject noise
    if genNoise == 'train':
        x_train = generateNoise(x_train, rc, rn)
    elif genNoise == 'test':
        x_test = generateNoise(x_test, rc, rn)

    for iter in range(runn):
        # Fit a simple decision tree first
        clf_tree = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, random_state=1)
        er_tree = generic_clf(y_train, x_train, y_test, x_test, clf_tree)

        # Fit Adaboost classifier using a decision tree stump
        er_train, er_test, = [er_tree[0]], [er_tree[1]]
        op = adaboost_clf(y_train, x_train, y_test, x_test, 11, clf_tree, er_train, er_test)
        er_tree_list_train.append(op[0])
        er_tree_list_test.append(op[1])
        y_pred_list_train.append(op[2])
        y_pred_list_test.append(op[3])
        estimator_list_train.append(op[4])
        estimator_list_test.append(op[5])
        estimator_error_list.append(op[6])
        alphaList.append(op[7])
        weight_list.append(op[8])
        trees_list.append(op[9])

        # Convert to np array for convenience
        er_tree_list_traiN = np.asarray(er_tree_list_train)
        er_tree_list_tesT = np.asarray(er_tree_list_test)

        estimator_list_traiN = np.asarray(estimator_list_train)
        estimator_list_tesT = np.asarray(estimator_list_test)

        y_predict_list_traiN = np.asarray(y_pred_list_train)
        y_predict_list_tesT = np.asarray(y_pred_list_test)
        y_predict_list_tesT.sum()

        estimator_error_lisT = np.asarray(estimator_error_list)

        alphaLisT = np.asarray(alphaList)

        weight_lisT = np.asarray(weight_list)
        trees_lisT = np.asarray(trees_list)

    trees_lisT = np.asarray(trees_list)
    alphaLisT = np.asarray(alphaList)
    estimator_list_traiN = np.asarray(estimator_list_train)

    pred_yTest = list()
    for item in x_test:
        accuracy = 0
        for li in range(estimator_list_traiN.shape[0]):
            for lii in range(estimator_list_traiN.shape[1]):
                accuracy += trees_lisT[li][lii].predict(item.reshape(1, item.shape[0])) * alphaLisT[li][lii]
        if accuracy >= 0:
            pred_yTest.append(1)
        else:
            pred_yTest.append(-1)

    # minor class reports
    convert_ypred = 1 * (pred_yTest == 1)
    convert_y_2to1 = 1 * (y_test == 1)
    report = prfs(y_test, pred_yTest, average='binary')
    fpr, tpr, thresh = roc_curve(y_test, pred_yTest)
    report_auc = auc(fpr, tpr)
    report_accuracy = get_acc_rate(pred_yTest, y_test) * 100
    gmean = np.sqrt(np.multiply(tpr, fpr)) * 100

    precision_list.append(report[0])
    recall_list.append(report[1])
    f1_list.append(report[2])
    auc_list.append(report_auc)
    accuracy_list.append(report_accuracy)
    g_mean_list.append(gmean[1])
    print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
        report[0] * 100,
        report[1] * 100,
        report[2] * 100,
        report_auc,
        report_accuracy,
        gmean[1]))

    # preds = list()
    # # Predictions Train
    # for li in range(estimator_list_traiN.shape[0]):
    #     for lii in range(estimator_list_traiN.shape[1]):
    #         preds[li][lii] = (np.array([np.sign((y_predict_list_traiN[li][lii][point] * weight_lisT[li][lii]).sum()) for point in range(y_predict_list_traiN.shape[2])]))
    #         print('Accuracy = ', (preds[li][lii] == y_train.values).sum() / y_predict_list_traiN.shape[2])

    # print("train acc",er_tree_list_traiN)
    # print("test acc",er_tree_list_tesT)

    # # Predictions Test
    # preds = (np.array([np.sign((y_predict_list[:, point] * estimator_weight_list).sum()) for point in range(N)]))
    # print('Accuracy = ', (preds == y).sum() / N)
    return precision_list, recall_list, f1_list, auc_list, accuracy_list, g_mean_list
    # plot_error_rate(er_train, er_test)


def run(times, runn,rc,rn,genNoise):
    for tim in range(times):
        precision, recall, f1, auc, accuracy, g_mean = adaboostUnderSampling('Covid-19.csv', runn,rc,rn,genNoise)
        print("============================= mean set: =============================")
        print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}'.format(
            np.mean(precision),
            np.mean(recall),
            np.mean(f1),
            np.mean(auc),
            np.mean(accuracy),
            np.mean(g_mean)))
        print("============================= std set: =============================")
        print('Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4}, G-mean = {5}\n\n'.format(
            np.std(precision),
            np.std(recall),
            np.std(f1),
            np.std(auc),
            np.std(accuracy),
            np.std(g_mean)))


def main():
    sets = [11, 31, 51, 101]
    # for noise
    rc = [0.1, 0.3, 0.5]
    rn = [0.1, 0.3, 0.5]
    runner = 10
    print("\n\n\n\n for train")
    for sett in sets:
        for rcc in rc:
            for rnn in rn:
                print('\n\ngenerate noise with rc = {0} and rn = {1} for only noisy train data'.format(rcc, rnn))
                print("\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(sett))
                run(runner, sett,rcc,rnn,genNoise='train')
    print("\n\n\n\n for test")
    for sett in sets:
        for rcc in rc:
            for rnn in rn:
                print('\n\ngenerate noise with rc = {0} and rn = {1} for only noisy test data'.format(rcc, rnn))
                print("\n+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(sett))
                run(runner, sett,rcc,rnn,genNoise='test')


# Driver
if __name__ == '__main__':
    main()
