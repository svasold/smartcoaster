import utils
import sys
from sklearn import svm

def svmClassifier(degree, plot_cm=True, print_accuracy=True, dimensions=None, export_model = False):
    """
    Fetches the data from the csv file and splits it into training and test data and projects it using either pca or lda
    (outsourced in classify function). Classifies the data using the svm classifier, optionally plots a confusion matrix,
    prints the accuracies and exports the model. Mean drink classification accuracy is returned.
    @param degree: degree of decision boundary polynomial
    @param plot_cm: flag for confusion matrix plotting
    @param print_accuracy: flag for printing of accuracies
    @param dimensions: number of dimensions to project on
    @param export_model: flag for exporting the model
    @return: mean over all drink classification accuracies
    """
    model = svm.SVC(kernel='poly', class_weight='balanced', degree=degree)
    if export_model:
        model_name = utils.dim_reduction_method + '_' + 'svm_' + str(dimensions) + '_dim_degree_' + str(degree) + \
                     '_classifier_' + utils.container + '.joblib'
        utils.exportModel(model, model_name, 'svm', dimensions)
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
    dimensions = int(sys.argv[5])
    degree = int(sys.argv[6])
    # '/home/marco/Schreibtisch/TU/Bac/data_spec/test_data_setup1.csv'
    utils.readCsv(csv_file_path)
    svmClassifier(degree=degree, dimensions=dimensions, export_model=True)
