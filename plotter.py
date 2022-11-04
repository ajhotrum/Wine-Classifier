import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_svm(rbf_path, linear_path, dataset,name):

    print("to")
    data_rbf = pd.read_csv(rbf_path)
    data_linear = pd.read_csv(linear_path)

    title = "{} SVMs".format(name)
    x_scale = 'log'
    y_scale='linear'
    x_label = 'Max Iteration'
    y_label = 'Accuracy (0.0 - 1.0)'

    train_points_rbf = data_rbf['train acc']
    test_points_rbf = data_rbf['test acc']
    train_points_linear = data_linear['train acc']
    test_points_linear = data_linear['test acc']
    train_sizes = data_rbf['param_SVM__max_iter']

    plt.close()
    plt.figure()
    plt.title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()


    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    plt.plot(train_sizes, train_points_linear, 'o-', linewidth=1, markersize=4,
             label="Linear Training Score")
    plt.plot(train_sizes, test_points_linear, 'o-', linewidth=1, markersize=4,
             label="Linear Cross-validation Score")

    plt.plot(train_sizes, train_points_rbf, 'o-', linewidth=1, markersize=4,
             label="RBF Training Score")
    plt.plot(train_sizes, test_points_rbf, 'o-', linewidth=1, markersize=4,
             label="RBF Cross-validation Score")

    plt.legend(loc="best")

    plt.savefig('output/images/{}_SVMs.png'.format(dataset), format='png', dpi=150)


def plot_ann(ann, ann_of, dataset,name):
    data_ann = pd.read_csv(ann)
    data_ann_of = pd.read_csv(ann_of)

    title = "{} ANNs".format(name)
    x_scale = 'log'
    y_scale='linear'
    x_label = 'Max Iteration'
    y_label = 'Accuracy (0.0 - 1.0)'

    train_points_ann = data_ann['train acc']
    test_points_ann = data_ann['test acc']
    train_points_ann_of = data_ann_of['train acc']
    test_points_ann_of = data_ann_of['test acc']
    train_sizes = data_ann_of['param_MLP__max_iter']

    plt.close()
    plt.figure()
    plt.title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()


    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    plt.plot(train_sizes, train_points_ann, 'o-', linewidth=1, markersize=4,
             label="ANN Training Score")
    plt.plot(train_sizes, test_points_ann, 'o-', linewidth=1, markersize=4,
             label="ANN Cross-validation Score")

    plt.plot(train_sizes, train_points_ann_of, 'o-', linewidth=1, markersize=4,
             label="ANN Training Score (No Reg)")
    plt.plot(train_sizes, test_points_ann_of, 'o-', linewidth=1, markersize=4,
             label="ANN Cross-validation Score (No Reg)")

    plt.legend(loc="best")

    plt.savefig('output/images/{}_ANNs.png'.format(dataset), format='png', dpi=150)


def plot_LC(ann, boost, dt, knn, svm_rbf, svm_linear, dataset, name):
    data_ann = pd.read_csv(ann)
    data_boost = pd.read_csv(boost)
    data_dt = pd.read_csv(dt)
    data_knn = pd.read_csv(knn)
    data_svm_rbf = pd.read_csv(svm_rbf)
    data_svm_linear = pd.read_csv(svm_linear)

    train_sizes = data_ann.iloc[:,0]
    test_points_ann = np.mean(data_ann.iloc[:,1:], axis = 1)
    test_points_boost = np.mean(data_boost.iloc[:,1:], axis = 1)
    test_points_dt = np.mean(data_dt.iloc[:,1:], axis = 1)
    test_points_knn = np.mean(data_knn.iloc[:,1:], axis = 1)
    test_points_svm_rbf = np.mean(data_svm_rbf.iloc[:,1:], axis = 1)
    test_points_svm_linear = np.mean(data_svm_linear.iloc[:,1:], axis = 1)


    title = "{} Learning Curves".format(name)
    x_scale = 'linear'
    y_scale='linear'
    x_label = 'Training examples (count)'
    y_label = 'Accuracy (0.0 - 1.0)'


    plt.close()
    plt.figure()
    plt.title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()


    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    plt.plot(train_sizes, test_points_ann, 'o-', linewidth=1, markersize=4,
             label="ANN")
    plt.plot(train_sizes, test_points_boost, 'o-', linewidth=1, markersize=4,
             label="Boosting")
    plt.plot(train_sizes, test_points_dt, 'o-', linewidth=1, markersize=4,
             label="DT")
    plt.plot(train_sizes, test_points_knn, 'o-', linewidth=1, markersize=4,
             label="KNN")
    plt.plot(train_sizes, test_points_svm_rbf, 'o-', linewidth=1, markersize=4,
             label="SVM (RBF)")
    plt.plot(train_sizes, test_points_svm_linear, 'o-', linewidth=1, markersize=4,
             label="SVM (Linear)")

    plt.legend(loc="best")

    plt.savefig('output/images/{}_LC.png'.format(dataset), format='png', dpi=150)


def plot_timing(ann, boost, dt, knn, svm_rbf, svm_linear, dataset, name):
    data_ann = pd.read_csv(ann)
    data_boost = pd.read_csv(boost)
    data_dt = pd.read_csv(dt)
    data_knn = pd.read_csv(knn)
    data_svm_rbf = pd.read_csv(svm_rbf)
    data_svm_linear = pd.read_csv(svm_linear)

    train_sizes = data_ann.iloc[:,0]
    test_points_ann = data_ann['test']
    train_points_ann = data_ann['train']
    test_points_boost = data_boost['test']
    train_points_boost = data_boost['train']
    test_points_dt = data_dt['test']
    train_points_dt = data_dt['train']
    test_points_knn = data_knn['test']
    train_points_knn = data_knn['train']
    test_points_svm_rbf = data_svm_rbf['test']
    train_points_svm_rbf = data_svm_rbf['train']
    test_points_svm_linear = data_svm_linear['test']
    train_points_svm_linear = data_svm_linear['train']



    title = "{} Fit Timing Curves".format(name)
    x_scale = 'linear'
    y_scale='linear'
    x_label = 'Training Data Size (% of Total)'
    y_label = 'Time (s)'


    plt.close()
    plt.figure()
    plt.title(title)
    plt.ylim(0,1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()


    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    plt.plot(train_sizes, train_points_ann, 'o-', linewidth=1, markersize=4,
             label="ANN")
    plt.plot(train_sizes, train_points_boost, 'o-', linewidth=1, markersize=4,
             label="Boosting")
    plt.plot(train_sizes, train_points_dt, 'o-', linewidth=1, markersize=4,
             label="DT")
    plt.plot(train_sizes, train_points_knn, 'o-', linewidth=1, markersize=4,
             label="KNN")
    plt.plot(train_sizes, train_points_svm_rbf, 'o-', linewidth=1, markersize=4,
             label="SVM (RBF)")
    plt.plot(train_sizes, train_points_svm_linear, 'o-', linewidth=1, markersize=4,
             label="SVM (Linear)")


    plt.legend(loc="best")
    plt.savefig('output/images/{}_fit_timing.png'.format(dataset), format='png', dpi=150)



    plt.close()
    plt.figure()
    title = "{} Predict Timing Curves".format(name)
    plt.title(title)
    #plt.ylim(0,0.5)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()


    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)
    plt.plot(train_sizes, test_points_ann, 'o-', linewidth=1, markersize=4,
             label="ANN")
    plt.plot(train_sizes, test_points_boost, 'o-', linewidth=1, markersize=4,
             label="Boosting ")
    plt.plot(train_sizes, test_points_dt, 'o-', linewidth=1, markersize=4,
             label="DT")
    plt.plot(train_sizes, test_points_knn, 'o-', linewidth=1, markersize=4,
             label="KNN")
    plt.plot(train_sizes, test_points_svm_rbf, 'o-', linewidth=1, markersize=4,
         label="SVM (RBF)")
    plt.plot(train_sizes, test_points_svm_linear, 'o-', linewidth=1, markersize=4,
         label="SVM (Linear)")

    plt.legend(loc="best")
    plt.savefig('output/images/{}_predict_timing.png'.format(dataset), format='png', dpi=150)

def run_plotter():
    print("OH HEYYYYY")
    dummy = input()

    # SVM
    rbf_path = "output/ITERtestSET_SVM_RBF_white_wine_data.csv"
    linear_path= 'output/ITERtestSET_SVMLinear_white_wine_data.csv'
    data = 'White_Wine_Quality'
    name = 'White Wine Quality'

    plot_svm(rbf_path, linear_path, data, name)

    rbf_path = "output/ITERtestSET_SVM_RBF_hill_valley.csv"
    linear_path= 'output/ITERtestSET_SVMLinear_hill_valley.csv'
    data = 'Hill_Valley'
    name = 'Hill-Valley'

    plot_svm(rbf_path, linear_path, data, name)

    # ANN
    ann_path = "output/ITERtestSET_ANN_white_wine.csv"
    ann_of_path= 'output/ITERtestSET_ANN_OF_white_wine.csv'
    data = 'White_Wine_Quality'
    name = 'White Wine Quality'

    plot_ann(ann_path, ann_of_path, data, name)

    ann_path = "output/ITERtestSET_ANN_hill_valley.csv"
    ann_of_path= 'output/ITERtestSET_ANN_OF_hill_valley.csv'
    data = 'Hill_Valley'
    name = 'Hill-Valley'

    plot_ann(ann_path, ann_of_path, data, name)


    # LC
    ann = "output/ANN_hill_valley_LC_test.csv"
    boost = 'output/Boost_hill_valley_LC_test.csv'
    dt = "output/DT_hill_valley_LC_test.csv"
    knn = 'output/KNN_hill_valley_LC_test.csv'
    svm_rbf = 'output/SVM_RBF_hill_valley_LC_test.csv'
    svm_linear = 'output/SVMLinear_hill_valley_LC_test.csv'
    data = 'Hill_Valley'
    name = 'Hill-Valley'

    plot_LC(ann, boost, dt, knn, svm_rbf, svm_linear, data, name)


    ann = "output/ANN_white_wine_LC_test.csv"
    boost = 'output/Boost_white_wine_LC_test.csv'
    dt = "output/DT_white_wine_LC_test.csv"
    knn = 'output/KNN_white_wine_LC_test.csv'
    svm_rbf = 'output/SVM_RBF_white_wine_LC_test.csv'
    svm_linear = 'output/SVMLinear_white_wine_LC_test.csv'
    data = 'White_Wine_Quality'
    name = 'White Wine Quality'

    plot_LC(ann, boost, dt, knn, svm_rbf, svm_linear, data, name)

    # Timing
    ann = "output/ANN_hill_valley_timing.csv"
    boost = 'output/Boost_hill_valley_timing.csv'
    dt = "output/DT_hill_valley_timing.csv"
    knn = 'output/KNN_hill_valley_timing.csv'
    svm_rbf = 'output/SVM_RBF_hill_valley_timing.csv'
    svm_linear = 'output/SVMLinear_hill_valley_timing.csv'
    data = 'Hill_Valley'
    name = 'Hill-Valley'

    plot_timing(ann, boost, dt, knn, svm_rbf, svm_linear, data, name)


    # Timing
    ann = "output/ANN_white_wine_timing.csv"
    boost = 'output/Boost_white_wine_timing.csv'
    dt = "output/DT_white_wine_timing.csv"
    knn = 'output/KNN_white_wine_timing.csv'
    svm_rbf = 'output/SVM_RBF_white_wine_timing.csv'
    svm_linear = 'output/SVMLinear_white_wine_timing.csv'
    data = 'White_Wine_Quality'
    name = 'White Wine Quality'

    plot_timing(ann, boost, dt, knn, svm_rbf, svm_linear, data, name)


if __name__ == '__main__':
    run_plotter()
