import utils
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def daClassifier(da_classifier, dimensions, plot_cm=True, print_accuracy=True, export_model = False):
    """
    Fetches the data from the csv file and splits it into training and test data and projects it using either pca or lda
    (outsourced in classify function). Classifies the data using a da classifier (either lda or qda), optionally plots a
    confusion matrix, prints the accuracies and exports the model. Mean drink classification accuracy is returned.
    @param da_classifier: The used classifier either "qda" or "lda"
    @param dimensions: number of dimensions to project on
    @param plot_cm: flag for confusion matrix plotting
    @param print_accuracy: flag for printing of accuracies
    @param export_model: flag for exporting the model
    @return: mean over all drink classification accuracies
    """
    if da_classifier == "lda":
        model = LinearDiscriminantAnalysis()
    elif da_classifier == "qda":
        model = QuadraticDiscriminantAnalysis()
    if export_model:
        model_name = utils.dim_reduction_method + '_' + da_classifier + '_' + str(dimensions) + '_dim_classifier_' + \
                     utils.container + '.joblib'
        utils.exportModel(model, model_name, da_classifier, dimensions)
    predictions, ground_truth = utils.classify(model, dimensions=dimensions)
    if plot_cm:
        if utils.container:
            if utils.without_weight:
                title = "Confusion matrix using the " + da_classifier + " classifier (" + setup + ", " + utils.container + ", without weight as feature)"
            else:
                title = "Confusion matrix using the " + da_classifier + " classifier (" + setup + ", " + utils.container + ")"
        else:
            if utils.without_weight:
                title = "Confusion matrix using the " + da_classifier + " classifier (" + setup + ", all containers" + ", without weight as feature)"
            else:
                title = "Confusion matrix using the " + da_classifier + " classifier ( " + setup + ", all containers" + ")"
        utils.plotConfusionMatrix(predictions, ground_truth, title)
    return utils.calcAcuracy(predictions, ground_truth, printing=print_accuracy)

if __name__ == "__main__":
    global setup
    csv_file_path = sys.argv[1]
    setup = sys.argv[2]
    utils.container = sys.argv[3]
    utils.dim_reduction_method = sys.argv[4]
    dimensions = int(sys.argv[5])
    classifier = sys.argv[6]
    # '/home/marco/Schreibtisch/TU/Bac/data_spec/test_data_setup1.csv'
    utils.readCsv(csv_file_path)
    daClassifier(da_classifier=classifier, dimensions=dimensions, export_model=True)
