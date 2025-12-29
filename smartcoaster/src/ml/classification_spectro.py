import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import joblib

def calcAcuracy(predictions, targets, printing=False):
    """
    This function calculates the classification accuracies for each drink and prints them if printing is true.
    Additionally, it prints the overall accuracy in the sense of #correct predictions / #predictions and an overall
    accuracy in the sense of the average over all the individual drink classification accuracies. (the same if there are
    the same number of samples for each drink.
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
    mean = sum / accuracy_count
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

def knnOverNeighboursAndDimensions(neighbour_range, dimension_range):
    # PCA with knn, plot of accuracies (mean over a few iterations) for different neighbour counts and dimensions
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
    if container:
        if without_weight:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", "+container+", without weight as feature)")
        else:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", "+container+")")
    else:
        if without_weight:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", all containers, without weight as feature)")
        else:
            plt.title("Knn accuracy over different number of neighbours ("+setup+", all containers)")

    plt.savefig("knn_" + dim_reduction_method + "_" + str(dimension_range[accuracies_argmax[0]]) + "_" +
                str(neighbour_range[accuracies_argmax[1]]) + "_" + str(accuracies_max) + ".png")
    plt.show()

def knnClassifier(k, plot_cm=True, print_accuracy=True, dimensions=None):
    model = KNeighborsClassifier(n_neighbors=k)
    predictions, ground_truth = classify(model, dimensions=dimensions)
    if plot_cm:
        if container:
            if without_weight:
                title = "Confusion matrix using the knn classifier (" + str(k) + " neighbours, " + str(dimensions) \
                        + " dimensions, " + setup + ", " + container + ", without weight as feature)"
            else:
                title = "Confusion matrix using the knn classifier (" + str(
                    k) + " neighbours, " + str(dimensions) + " dimensions, " + setup + ", " + container +")"
        else:
            if without_weight:
                title = "Confusion matrix using the knn classifier (" + str(k) + " neighbours, " + str(dimensions) \
                        + " dimensions, " + setup + ", " + "all containers" + ", without weight as feature)"
            else:
                title = "Confusion matrix using the knn classifier (" + str(
                    k) + " neighbours, " + str(dimensions) + " dimensions, " + setup + ", " + "all containers" + ")"
        plotConfusionMatrix(predictions, ground_truth, title)
    return calcAcuracy(predictions, ground_truth, print_accuracy)

def daOverDimensions(da_classifier, dimension_range):
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
    if container:
        if without_weight:
            plt.title(da_classifier + " classifier accuracy over different number of components (" + setup + ", " +
                      container + ", without weight as feature)")
        else:
            plt.title(da_classifier + " classifier accuracy over different number of components (" + setup + ", " +
                      container + ")")
    else:
        if without_weight:
            plt.title(
                da_classifier + " classifier accuracy over different number of components (" + setup +
                ", all containers, without weight as feature)")
        else:
            plt.title(da_classifier + " classifier accuracy over different number of components (" + setup +
                      ", all containers)")

    plt.savefig(da_classifier + "_" + dim_reduction_method + "_" + str(dimension_range[accuracies_argmax]) + "_"
                + str(np.max(accuracies)) + ".png")
    plt.show()

def daClassifier(da_classifier, dimensions, plot_cm=True, print_accuracy=True):
    if da_classifier == "lda":
        model = LinearDiscriminantAnalysis()
    elif da_classifier == "qda":
        model = QuadraticDiscriminantAnalysis()
    predictions, ground_truth = classify(model, dimensions=dimensions)
    if plot_cm:
        if container:
            if without_weight:
                title = "Confusion matrix using the lda classifier (" + setup + ", " + container + ", without weight as feature)"
            else:
                title = "Confusion matrix using the lda classifier (" + setup + ", " + container + ")"
        else:
            if without_weight:
                title = "Confusion matrix using the lda classifier (" + setup + ", all containers" + ", without weight as feature)"
            else:
                title = "Confusion matrix using the lda classifier ( " + setup + ", all containers" + ")"
        plotConfusionMatrix(predictions, ground_truth, title)
    return calcAcuracy(predictions, ground_truth, printing=print_accuracy)

def svmOverDegreesAndDimensions(degree_range, dimension_range):
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
    if container:
        if without_weight:
            plt.title("Svm accuracy over different degrees of the decision boundary polynomial (" + setup + ", " +
                      container + ", without weight as feature)")
        else:
            plt.title("Svm accuracy over different degrees of the decision boundary polynomial (" + setup + ", " +
                      container + ")")
    else:
        if without_weight:
            plt.title(
                "Svm accuracy over different degrees of the decision boundary polynomial (" + setup +
                ", all containers, without weight as feature)")
        else:
            plt.title("Svm accuracy over different degrees of the decision boundary polynomial (" + setup +
                      ", all containers)")

    plt.savefig("svm_" + dim_reduction_method + "_" + str(dimension_range[accuracies_argmax[0]]) + "_" +
                str(degree_range[accuracies_argmax[1]]) + "_" + str(accuracies_max) + ".png")
    plt.show()

def svmClassifier(degree, plot_cm=True, print_accuracy=True, dimensions=None):
    model = svm.SVC(kernel='poly', class_weight='balanced', degree=degree)
    predictions, ground_truth = classify(model, dimensions=dimensions)
    if plot_cm:
        if container:
            if without_weight:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions)\
                        + " dimensions, " + setup + ", " + container + ", without weight as feature)"
            else:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions)\
                        + " dimensions, " + setup + ", " + container + ")"
        else:
            if without_weight:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions) \
                        + " dimensions, " + setup + ", " + "all containers" + ", without weight as feature)"
            else:
                title = "Confusion matrix using the svm classifier (degree " + str(degree) + ", " + str(dimensions)\
                        + " dimensions, " + setup + ", " + "all containers" + ")"
        plotConfusionMatrix(predictions, ground_truth, title)
    return calcAcuracy(predictions, ground_truth, print_accuracy)

#todo add function headers for those 3 functions
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


def classify(model, dimensions=None):
    """
    This function gets the training/test data splits, optionally projects the samples into a lower dimensional space using
    either pca or lda (global variable dim_reduction_method), fits a model (passed as parameter) and classifies the test samples. The predictions and the ground
    truth values get returned.
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
                        reduction_method.fit(training_samples_split, training_targets[split_index])  # lda
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


def visualizeData(scaling=False, three_dimensional=False):
    """
    Visualizes the samples in a 2d or 3d space, either projected using pca or lda, with or without weight as feature.
    The samples get colored according to their respective drink to visualize the class distribution.
    @param scaling: Flag for scaling
    @param three_dimensional: Flag for 3d projection
    """
    # discard every fifth sample
    indices = np.arange(len(csv_data))
    data = csv_data[indices % 5 != 0]
    # choose from specified container
    if container:
        data = data[data[:, 2] == container]
    # split in features and targets
    if without_weight:
        features = data[:, csv_columns[0:-2]]
    else:
        features = data[:, csv_columns[0:-1]]
    targets = data[:, csv_columns[-1]]
    weights = data[:, csv_columns[-2]].astype(float)
    if scaling:
        features = min_max_scaler.fit_transform(features)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#fc8c03', '#fc038c']
    if dim_reduction_method == "pca":
        pca = PCA(n_components=2)
        pca.fit(features)
        projected_features = pca.transform(features)
    elif dim_reduction_method == "lda":
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(features, targets)
        projected_features = lda.transform(features)

    fig = plt.figure(figsize=(7, 7))
    if three_dimensional:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    for i, drink in enumerate(drinks):
        selected_features = projected_features[targets == drink]
        if three_dimensional:
            ax.scatter(selected_features[:, 0], selected_features[:, 1], weights[targets == drink], c=colors[i], label=drink)
        else:
            ax.scatter(selected_features[:, 0], selected_features[:, 1], c=colors[i], label=drink)
    if container:
        if without_weight:
            title = "Visualization of the data projected into a 2d space using " + dim_reduction_method + " (" + setup + ", " + container + ", without weight as feature)"
        else:
            title = "Visualization of the data projected into a 2d space using " + dim_reduction_method + " (" + setup + ", " + container + ")"
    else:
        if without_weight:
            title = "Visualization of the data projected into a 2d space using " + dim_reduction_method + " (" + setup + ", all containers, without weight as feature)"
        else:
            title = "Visualization of the data projected into a 2d space using " + dim_reduction_method + " (" + setup + ", all containers)"
    plt.legend(loc="best")
    plt.title(title)
    plt.show()

def loadModelCLassify():
    model = joblib.load('pca_knn_13_dim_5_neighbours_classifier_glass.joblib')
    reduction_method = joblib.load('pca_knn_13_dim_projection_glass.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    if container:
        data = csv_data[csv_data[:, 2] == container]
    else:
        data = csv_data
    # discard every fifth sample
    indices = np.arange(len(data))
    data = data[indices % 5 != 0]
    features = data[:, csv_columns[0:-1]]
    targets = data[:, csv_columns[-1]]
    features = scaler.transform(features)
    features = reduction_method.transform(features)
    predictions = model.predict(features)
    print(predictions)
    calcAcuracy(predictions, targets, printing=True)

def main():
    # visualizeData(scaling=True, three_dimensional=False)
    # knnOverNeighboursAndDimensions(neighbour_range=range(1, 20, 2), dimension_range=range(1, 25, 2)) #lda range(1, len(drinks)) pca range(1, 26, 2)
    # knnClassifier(k=19, dimensions=17)
    # svmOverDegreesAndDimensions(degree_range=range(1, 2), dimension_range=range(1, len(drinks)))
    # svmClassifier(degree=1, dimensions=11)
    # daOverDimensions(da_classifier="qda", dimension_range=range(1, 26, 2))
    # daOverDimensions(da_classifier="qda", dimension_range=range(1, len(drinks)))
    loadModelCLassify()
    # neural()

    # for con in ["glass", "mug", "wide glass"]:
    #     global container
    #     global dim_reduction_method
    #     container = con
    #     print("container", container)
    #     dim_reduction_method = "pca"
    #     knnOverNeighboursAndDimensions(neighbour_range=range(1, 20, 2), dimension_range=range(1, 26, 2))
    #     svmOverDegreesAndDimensions(degree_range=range(1, 6), dimension_range=range(1, 26, 2))
    #     daOverDimensions(da_classifier="lda", dimension_range=range(1, 26, 2))
    #     daOverDimensions(da_classifier="qda", dimension_range=range(1, 26, 2))
    #     dim_reduction_method = "lda"
    #     knnOverNeighboursAndDimensions(neighbour_range=range(1, 20, 2), dimension_range=range(1, len(drinks)))
    #     svmOverDegreesAndDimensions(degree_range=range(1, 6), dimension_range=range(1, len(drinks)))
    #     daOverDimensions(da_classifier="lda", dimension_range=range(1, len(drinks)))
    #     daOverDimensions(da_classifier="qda", dimension_range=range(1, len(drinks)))
    print("finish")


# global variables
setup = "setup_1"
container = "glass" #none, glass, mug, wide glass
without_weight = False
dim_reduction_method = "lda"
# Read Data
with open('/home/marco/Schreibtisch/TU/Bac/data_spec/test_data_setup1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    csv_data = np.array(list(reader))
drinks, counts = np.unique(csv_data[:, 0], return_counts=True)
print(dict(zip(drinks, counts)))
# csv_columns = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 1, 0]
csv_columns = list(range(5, csv_data.shape[1]))
csv_columns.extend([1, 0])

min_max_scaler = preprocessing.MinMaxScaler()
k_fold = KFold(n_splits=5)
ITERATIONS = 10

if __name__ == "__main__":
    main()







# logistic regression
# logisticRegr = LogisticRegression(solver = 'lbfgs')
# logisticRegr.fit(training_samples, training_targets)
# logisticRegr.predict(test_samples)
# subset_accuracy = logisticRegr.score(test_samples, test_targets)
# print("subset accuracy:", subset_accuracy)

# logisticRegr.predict(test_img[0:10]), 10






