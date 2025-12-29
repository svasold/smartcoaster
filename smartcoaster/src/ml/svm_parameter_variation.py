import utils
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def svmOverDegreesAndDimensions(degree_range, dimension_range):
    """
    Projects the data onto a range of dimensions and varying the decision boundary polynomial degree. Plots a plot with
    one graph for each dimensionality representing the mean classification accuracy over decision boundary polynomials
    of different degrees. Prints the maximum accuracy and the respective configuration.
    @param degree_range: range object containing the different polynomial degrees
    @param dimension_range: range object containing the different dimension counts
    """
    accuracies = []
    for dim in dimension_range:
        print("dimensions:", dim)
        for degree in degree_range:
            print("degree", degree)
            accuracies.append(svmClassifier(degree, plot_cm=False, print_accuracy=False, dimensions=dim))
    accuracies = np.asarray(accuracies).reshape(len(dimension_range), len(degree_range))
    accuracies_argmax = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    accuracies_max = accuracies[accuracies_argmax]
    print("max accuracy " + str(accuracies_max) + " is achieved with " + str(dimension_range[accuracies_argmax[0]]) +
          " dimensions and a degree of " + str(degree_range[accuracies_argmax[1]]))
    # plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for index, accuracy in enumerate(accuracies):
        ax.plot(degree_range, accuracy, label=str(dimension_range[index]) + " components")
    plt.ylabel('accuracy')
    plt.xlabel('degree')
    plt.legend()
    if utils.container:
        if utils.without_weight:
            plt.title("Svm accuracy over different degrees of the decision boundary polynomial (" + setup + ", " +
                      utils.container + ", without weight as feature)")
        else:
            plt.title("Svm accuracy over different degrees of the decision boundary polynomial (" + setup + ", " +
                      utils.container + ")")
    else:
        if utils.without_weight:
            plt.title(
                "Svm accuracy over different degrees of the decision boundary polynomial (" + setup +
                ", all containers, without weight as feature)")
        else:
            plt.title("Svm accuracy over different degrees of the decision boundary polynomial (" + setup +
                      ", all containers)")

    plt.savefig("svm_" + utils.dim_reduction_method + "_" + str(dimension_range[accuracies_argmax[0]]) + "_" +
                str(degree_range[accuracies_argmax[1]]) + "_" + str(accuracies_max) + ".png")
    plt.show()

def svmClassifier(degree, plot_cm=True, print_accuracy=True, dimensions=None):
    """
    Fetches the data from the csv file and splits it into training and test data and projects it using either pca or lda
    (outsourced in classify function). Classifies the data using the svm classifier, optionally plots a confusion matrix
    and prints the accuracies. Mean drink classification accuracy is returned.
    @param degree: degree of decision boundary polynomial
    @param plot_cm: flag for confusion matrix plotting
    @param print_accuracy: flag for printing of accuracies
    @param dimensions: number of dimensions to project on
    @return: mean over all drink classification accuracies
    """
    model = svm.SVC(kernel='poly', class_weight='balanced', degree=degree)
    predictions, ground_truth = utils.classify(model, dimensions=dimensions)
    if plot_cm:
        if utils.container:
            if utils.without_weight:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions)\
                        + " dimensions, " + setup + ", " + utils.container + ", without weight as feature)"
            else:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions)\
                        + " dimensions, " + setup + ", " + utils.container + ")"
        else:
            if utils.without_weight:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions) \
                        + " dimensions, " + setup + ", " + "all containers" + ", without weight as feature)"
            else:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions)\
                        + " dimensions, " + setup + ", " + "all containers" + ")"
        utils.plotConfusionMatrix(predictions, ground_truth, title)
    return utils.calcAcuracy(predictions, ground_truth, print_accuracy)


if __name__ == "__main__":
    global setup
    csv_file_path = sys.argv[1]
    setup = sys.argv[2]
    utils.container = sys.argv[3]
    utils.dim_reduction_method = sys.argv[4]
    dimension_range = range(int(sys.argv[5]), int(sys.argv[6]) + 1, int(sys.argv[7]))
    degree_range = range(int(sys.argv[8]), int(sys.argv[9]) + 1, int(sys.argv[10]))
    # '/home/marco/Schreibtisch/TU/Bac/data_spec/test_data_setup1.csv'
    utils.readCsv(csv_file_path)
    svmOverDegreesAndDimensions(dimension_range=dimension_range, degree_range=degree_range)
