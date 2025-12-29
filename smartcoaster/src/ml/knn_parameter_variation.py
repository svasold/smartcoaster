import sys
import utils
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import matplotlib.pyplot as plt

def knnClassifier(k, plot_cm=True, print_accuracy=True, dimensions=None):
    """
    Fetches the data from the csv file and splits it into training and test data and projects it using either pca or lda
    (outsourced in classify function). Classifies the data using the knn classifier, optionally plots a confusion matrix
    and prints the accuracies. Mean drink classification accuracy is returned.
    @param k: number of neighbours
    @param plot_cm: flag for confusion matrix plotting
    @param print_accuracy: flag for printing of accuracy
    @param dimensions: number of dimensions
    @return: mean over all drink classification accuracies
    """
    model = KNeighborsClassifier(n_neighbors=k)
    predictions, ground_truth = utils.classify(model, dimensions=dimensions)
    if plot_cm:
        if utils.container:
            if utils.without_weight:
                title = "Confusion matrix using the knn classifier (" + str(k) + " neighbours, " + str(dimensions) \
                        + " dimensions, " + setup + ", " + utils.container + ", without weight as feature)"
            else:
                title = "Confusion matrix using the knn classifier (" + str(
                    k) + " neighbours, " + str(dimensions) + " dimensions, " + setup + ", " + utils.container +")"
        else:
            if utils.without_weight:
                title = "Confusion matrix using the knn classifier (" + str(k) + " neighbours, " + str(dimensions) \
                        + " dimensions, " + setup + ", " + "all containers" + ", without weight as feature)"
            else:
                title = "Confusion matrix using the knn classifier (" + str(
                    k) + " neighbours, " + str(dimensions) + " dimensions, " + setup + ", " + "all containers" + ")"
        utils.plotConfusionMatrix(predictions, ground_truth, title)
    return utils.calcAcuracy(predictions, ground_truth, print_accuracy)
def knnOverNeighboursAndDimensions(neighbour_range, dimension_range):
    """
    Projects the data onto a range of dimensions and varying the neighbour count. Plots a plot with one graph for each
    dimensionality representing the mean classification accuracy over different neighbour counts. Prints the maximum
    accuracy and the respective configuration.
    @param neighbour_range: range object containing the different neighbour counts
    @param dimension_range: range object containing the different dimension counts
    """
    accuracies = []
    for dim in dimension_range:
        print(str(dim) + " dimensions")
        for k in neighbour_range:
            accuracies.append(knnClassifier(k, plot_cm=False, print_accuracy=False, dimensions=dim))
    accuracies = np.asarray(accuracies).reshape(len(dimension_range), len(neighbour_range))
    accuracies_argmax = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    accuracies_max = accuracies[accuracies_argmax]
    print("max accuracy " + str(accuracies_max) + " is achieved with " + str(dimension_range[accuracies_argmax[0]]) +
          " dimensions and " + str(neighbour_range[accuracies_argmax[1]]) + " neighbours")
    # plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for index, accuracy in enumerate(accuracies):
        ax.plot(neighbour_range, accuracy, label=str(dimension_range[index]) + " components")
    plt.ylabel('accuracy')
    plt.xlabel('neighbours')
    plt.legend()
    if utils.container:
        if utils.without_weight:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", "+utils.container+", without weight as feature)")
        else:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", "+utils.container+")")
    else:
        if utils.without_weight:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", all containers, without weight as feature)")
        else:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", all containers)")

    plt.savefig("knn_" + utils.dim_reduction_method + "_" + str(dimension_range[accuracies_argmax[0]]) + "_" +
                str(neighbour_range[accuracies_argmax[1]]) + "_" + str(accuracies_max) + ".png")
    plt.show()

if __name__ == "__main__":
    global setup
    csv_file_path = sys.argv[1]
    setup = sys.argv[2]
    utils.container = sys.argv[3]
    utils.dim_reduction_method = sys.argv[4]
    dimension_range = range(int(sys.argv[5]), int(sys.argv[6]) + 1, int(sys.argv[7]))
    neighbour_range = range(int(sys.argv[8]), int(sys.argv[9]) + 1, int(sys.argv[10]))

    # '/home/marco/Schreibtisch/TU/Bac/data_spec/test_data_setup1.csv'
    utils.readCsv(csv_file_path)
    knnOverNeighboursAndDimensions(neighbour_range=neighbour_range, dimension_range=dimension_range)

