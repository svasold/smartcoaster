import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import csv
import joblib

# global variables
container = None #none, glass, mug, wide glass
without_weight = False
dim_reduction_method = None
drinks = []
csv_data = None
csv_columns = []
min_max_scaler = preprocessing.MinMaxScaler()
k_fold = KFold(n_splits=5)
ITERATIONS = 10


def classify(model, dimensions=None):
    """
    This function gets the training/test data splits, optionally projects the samples into a lower dimensional space using
    either pca or lda (global variable dim_reduction_method), fits a model (passed as parameter) and classifies the test
    samples. The predictions and the ground truth values get returned.
    @param model: The sklearn machine learning model (either knn, svm, lda or qda)
    @param dimensions: Number of dimension on which to project on (no projection if set to None)
    @return: models predictions and the ground truth target values
    """
    predictions = []
    test_target_list = []
    if dim_reduction_method == "pca":
        reduction_method = PCA(n_components=dimensions)
    elif dim_reduction_method == "lda":
        reduction_method = LinearDiscriminantAnalysis(n_components=dimensions)
    for iteration in range(0, ITERATIONS):
        training_samples, training_targets, test_samples, test_targets = getSamples(scaling=True, container=container)
        for split_index, training_samples_split in enumerate(training_samples):
            if dimensions is not None:
                if dim_reduction_method == "pca":
                    reduction_method.fit(training_samples_split)
                elif dim_reduction_method == "lda":
                    try:
                        reduction_method.fit(training_samples_split, training_targets[split_index])
                    except ValueError:
                        continue
                training_samples_split = reduction_method.transform(training_samples_split)
                test_samples_split = reduction_method.transform(test_samples[split_index])
            else:
                test_samples_split = test_samples[split_index]

            model.fit(training_samples_split, training_targets[split_index])
            predictions.extend(model.predict(test_samples_split))
            test_target_list.extend(test_targets[split_index])
    return predictions, test_target_list

def plotConfusionMatrix(predictions, ground_truth, title):
    """
    Plots a confusion matrix given the predictions and the ground truth targets.
    @param predictions: predictions of the model
    @param ground_truth: ground truth targets
    @param title: title of the confusion matrix
    """
    disp = ConfusionMatrixDisplay.from_predictions(ground_truth, predictions, labels=drinks, display_labels=drinks,
                                                   normalize='true', xticks_rotation='vertical')
    fig = disp.figure_
    fig.set_figwidth(10)
    fig.set_figheight(9)
    fig.suptitle(title)
    plt.show()

def calcAcuracy(predictions, targets, printing=False):
    """
    This function calculates the classification accuracies for each drink and prints them if printing is true.
    Additionally, it prints the overall accuracy in the sense of #correct predictions / #predictions and an overall
    accuracy in the sense of the average over all the individual drink classification accuracies. (the same if there are
    the same number of samples for each drink)
    @param predictions: predicted drinks
    @param targets: ground truth drinks
    @param printing: flag for printing the results
    @return: mean over all individual drink classification accuracies
    """
    accuracies = {}
    for drink in drinks:
        accuracies[drink] = [0, 0]
    sum = 0
    accuracy_count = 0

    for index, target in enumerate(targets):
        accuracies[target][0] += 1
        if target == predictions[index]:
            accuracies[target][1] += 1

    for drink, counts in accuracies.items():
        if counts[0] != 0:
            accuracy = counts[1] / counts[0]
            sum += accuracy
            accuracy_count += 1
        if printing:
            print(drink + ": " + str(accuracy))
    try:
        mean = sum / accuracy_count
    except ZeroDivisionError:
        mean = 0
    if printing:
        print("accuracy (correct predictions / all predictions):", metrics.accuracy_score(targets, predictions))
        print("mean accuracy:", mean)
    return mean

def getSamples(scaling=True, container=None):
    """
    This function extracts the needed samples out of the global variable csv_data which contains the whole csv file.
    When the parameter container is set, only samples of this container are selected. To select the right columns of the
    dataset, the global variable csv_columns is utilized. The data is randomly shuffled with every call to this function
    and it is split into training and test sets. K fold cross validation is used, so the return values have split_num
    entries. In the case of split_num = 5 (see global variable k_fold), 5 different training and testing samples/targets
    are returned. For example, 80 samples out of 100 are used as training samples, the split_num is 5 and 25 features
    are used, then the shape of the return value training_samples would be (5, 60, 25) .Additionally, the fetched data
    is scaled using a min max scaler. Samples corresponding to the same measurement (same drink and weight) are
    "bundled" together, so they get chosen in the same pool (either training or testing).
    @param scaling: Flag for scaling
    @param container: Either None, "glass", "mug" or "wide glass" to filter for container
    @return: The samples, split into training and test data (features and targets in seperate lists).
    """
    delta = 5
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    training_samples = []
    training_targets = []
    test_samples = []
    test_targets = []
    data = []
    twins = []
    # filter container
    if container:
        rows = csv_data[csv_data[:, 2] == container]
    else:
        rows = csv_data
    # discard every fifth sample
    indices = np.arange(len(rows))
    rows = rows[indices % 5 != 0]
    for row in rows[:, csv_columns]:
        if len(twins) == 0:
            twins.append(row)
        #similar weight and same drink
        elif int(row[-2]) - delta <= int(twins[0][-2]) <= int(row[-2]) + delta and row[-1] == twins[0][-1]:
            if twins[0][-1] == "beer" and len(twins) >= 20:
                data.append(np.asarray(twins))
                twins = [row]
                continue
            twins.append(row)
        else:
            data.append(np.asarray(twins))
            twins = [row]
    data.append(np.asarray(twins))
    np.random.shuffle(data)

    for train_indices, test_indices in k_fold.split(data):
        training_data_split = [data[i] for i in train_indices]
        test_data_split = [data[i] for i in test_indices]
        if without_weight:
            training_samples_split = np.asarray([item for sublist in training_data_split for item in sublist])[:, 0:-2]
            test_samples_split = np.asarray([item for sublist in test_data_split for item in sublist])[:, 0:-2]
        else:
            training_samples_split = np.asarray([item for sublist in training_data_split for item in sublist])[:, 0:-1]
            test_samples_split = np.asarray([item for sublist in test_data_split for item in sublist])[:, 0:-1]
        training_targets_split = np.asarray([item for sublist in training_data_split for item in sublist])[:, -1]
        test_targets_split = np.asarray([item for sublist in test_data_split for item in sublist])[:, -1]

        if scaling:
            training_samples_split = min_max_scaler.fit_transform(training_samples_split)
            test_samples_split = min_max_scaler.transform(test_samples_split)

        training_samples.append(training_samples_split)
        training_targets.append(training_targets_split)
        test_samples.append(test_samples_split)
        test_targets.append(test_targets_split)

    return training_samples, training_targets, test_samples, test_targets

def readCsv(csv_file_path):
    """
    Reads the csv file into the global variable csv_data und defines the relevant csv_columns which are used for training
    the model. Also, the selection of drinks is read out ot the csv file and stored in the global variable drinks.
    @param csv_file_path:
    """
    # Read Data
    global csv_data, csv_columns, drinks
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        csv_data = np.array(list(reader))
    # utils.csv_columns = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 1, 0]
    csv_columns = list(range(5, csv_data.shape[1]))
    csv_columns.extend([1, 0])
    drinks, counts = np.unique(csv_data[:, 0], return_counts=True)
    print(dict(zip(drinks, counts)))

def exportModel(model, model_name, classifier_name, dimensions=None, scaling=True):
    """
    This function optionally filters the dataset according to the container and projects it onto a lower dimensional
    space using either pca or lda. Then a model (specified through parameter model) is trained on the projected data.
    This model, the dimensionality reduction method and the scaler are exported as .joblib files.
    @param model: machine learning model to be trained
    @param model_name: model name for .joblib filename
    @param classifier_name: name of the classifier for .joblib filename
    @param dimensions: number of dimensions to project on
    @param scaling: flag for scaling
    """
    if container:
        data = csv_data[csv_data[:, 2] == container]
    else:
        data = csv_data
    # discard every fifth sample
    indices = np.arange(len(data))
    data = data[indices % 5 != 0]
    features = data[:, csv_columns[0:-1]]
    targets = data[:, csv_columns[-1]]
    if scaling:
        features = min_max_scaler.fit_transform(features)
    if dimensions:
        if dim_reduction_method == "pca":
            reduction_method = PCA(n_components=dimensions)
            reduction_method.fit(features)
        elif dim_reduction_method == "lda":
            reduction_method = LinearDiscriminantAnalysis(n_components=dimensions)
            reduction_method.fit(features, targets)
        features = reduction_method.transform(features)
        joblib.dump(reduction_method, dim_reduction_method + '_' + classifier_name + '_' + str(dimensions) +
                    '_dim_projection_' + container + '.joblib')
    model.fit(features, targets)
    joblib.dump(model, model_name)
    joblib.dump(min_max_scaler, 'min_max_scaler.joblib')

