import utils
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def daOverDimensions(da_classifier, dimension_range):
    """
    Plots the classification accuracy of the lda or qda classifier over different numbers of dimensions. Either pca or
    lda is used for projection.
    @param da_classifier: The used classifier either "qda" or "lda"
    @param dimension_range: range object containing dimensions for projection
    """
    accuracies = []
    for dim in dimension_range:
        print(str(dim) + " dimensions")
        accuracies.append(daClassifier(da_classifier, dim, False, False))
    accuracies = np.asarray(accuracies)
    accuracies_argmax = np.argmax(accuracies)
    print("max accuracy " + str(np.max(accuracies)) + " is achieved with " + str(dimension_range[accuracies_argmax]) +
          " dimensions")
    #plotting
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.plot(dimension_range, accuracies)
    plt.ylabel('accuracy')
    plt.xlabel('components')
    if utils.container:
        if utils.without_weight:
            plt.title(da_classifier + " classifier accuracy over different number of components (" + setup + ", " +
                      utils.container + ", without weight as feature)")
        else:
            plt.title(da_classifier + " classifier accuracy over different number of components (" + setup + ", " +
                      utils.container + ")")
    else:
        if utils.without_weight:
            plt.title(
                da_classifier + " classifier accuracy over different number of components (" + setup +
                ", all containers, without weight as feature)")
        else:
            plt.title(da_classifier + " classifier accuracy over different number of components (" + setup +
                      ", all containers)")

    plt.savefig(da_classifier + "_" + utils.dim_reduction_method + "_" + str(dimension_range[accuracies_argmax]) + "_"
                + str(np.max(accuracies)) + ".png")
    plt.show()

def daClassifier(da_classifier, dimensions, plot_cm=True, print_accuracy=True):
    """
    Fetches the data from the csv file and splits it into training and test data and projects it using either pca or lda
    (outsourced in classify function). Classifies the data using a da classifier (either lda or qda), optionally plots a
    confusion matrix and prints the accuracies. Mean drink classification accuracy is returned.
    @param da_classifier: The used classifier either "qda" or "lda"
    @param dimensions: number of dimensions to project on
    @param plot_cm: flag for confusion matrix plotting
    @param print_accuracy: flag for printing of accuracies
    @return: mean over all drink classification accuracies
    """
    if da_classifier == "lda":
        model = LinearDiscriminantAnalysis()
    elif da_classifier == "qda":
        model = QuadraticDiscriminantAnalysis()
    predictions, ground_truth = utils.classify(model, dimensions=dimensions)
    if plot_cm:
        if utils.container:
            if utils.without_weight:
                title = "Confusion matrix using the lda classifier (" + setup + ", " + utils.container + ", without weight as feature)"
            else:
                title = "Confusion matrix using the lda classifier (" + setup + ", " + utils.container + ")"
        else:
            if utils.without_weight:
                title = "Confusion matrix using the lda classifier (" + setup + ", all containers" + ", without weight as feature)"
            else:
                title = "Confusion matrix using the lda classifier ( " + setup + ", all containers" + ")"
        utils.plotConfusionMatrix(predictions, ground_truth, title)
    return utils.calcAcuracy(predictions, ground_truth, printing=print_accuracy)

if __name__ == "__main__":
    global setup
    csv_file_path = sys.argv[1]
    setup = sys.argv[2]
    utils.container = sys.argv[3]
    utils.dim_reduction_method = sys.argv[4]
    dimension_range = range(int(sys.argv[5]), int(sys.argv[6]) + 1, int(sys.argv[7]))
    classifier = sys.argv[8]
    # '/home/marco/Schreibtisch/TU/Bac/data_spec/test_data_setup1.csv'
    utils.readCsv(csv_file_path)
    daOverDimensions(dimension_range=dimension_range, da_classifier=classifier)
