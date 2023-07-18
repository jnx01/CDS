import random
import time
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def randomDataSplitting(normalized_dataset):

    # Set parameters
    num_subjects = 10
    num_train_images = 150
    num_test_images = 20
    num_data_splits = 5

    # Generate random data splits
    random_data_splits = []
    for _ in range(num_data_splits):
        data_split = random.sample(range(num_train_images + num_test_images), num_test_images)
        random_data_splits.append(data_split)


    # Split dataset into train and test sets for each random split
    train_sets = []
    test_sets = []

    train_labels = np.repeat(np.arange(num_subjects), num_train_images)       #Same label for same subject (regardless of split) hence need to build only one label set for training and test data
    test_labels = np.repeat(np.arange(num_subjects), num_test_images)

    for data_split in random_data_splits:
        train_set = []
        test_set = []

        for subject in range(num_subjects):
            start_index = subject * (num_train_images + num_test_images)
            end_index = start_index + (num_train_images + num_test_images)

            subject_data = normalized_dataset[start_index:end_index]
            # print(end_index)
            test_subject_data = subject_data[data_split]
            train_subject_data = np.delete(subject_data, data_split, axis=0)

            test_set.extend(test_subject_data)
            train_set.extend(train_subject_data)

        train_sets.append(train_set)
        test_sets.append(test_set)

    return train_sets, train_labels, test_sets, test_labels


def euclidean_distance(a, b):
    # Calculate the Euclidean distance between vectors a and b
    return np.sqrt(np.sum((a - b) ** 2))


def cosine_similarity(a, b):
    # Calculate the cosine similarity between vectors a and b
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return (1-similarity)


def k_nearest_neighbors(X_train, y_train, X_test, k, distance_measure):
    predictions = []

    for test_instance in X_test:
        distances = []

        for i in range(len(X_train)):
            train_instance = X_train[i]
            distance = distance_measure(test_instance, train_instance)
            distances.append((distance, y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        labels = [label for _, label in k_nearest]
        predicted_label = max(set(labels), key=labels.count)
        predictions.append(predicted_label)

    return predictions


def perform_pca(X, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(X)
    X_transformed = pca.transform(X)
    return pca, X_transformed


def evaluate_pca_knn(X_train, y_train, X_test, y_test, k, distance_measure, num_components):
    pca, X_train_transformed = perform_pca(X_train, num_components)
    X_test_transformed = pca.transform(X_test)
    predictions = k_nearest_neighbors(X_train_transformed, y_train, X_test_transformed, k, distance_measure)
    accuracy = np.mean(predictions == y_test)
    return accuracy




num_data_splits = 5


# Load dataset from CSV file
dataset = np.genfromtxt('/content/fea.csv', delimiter=',')
# dataset = dataset[:850] # Reduces data to 5 subjects
# dataset = dataset[:1190] # Reduces data to 7 subjects

# Preprocess dataset by normalizing vectors to unit length
normalized_dataset = normalize(dataset, norm='l2', axis=1)

# Build 5 random splits of the data
train_sets, train_labels, test_sets, test_labels = randomDataSplitting(normalized_dataset)

# Apply KNN and get results
accuracy_scores = []
computation_times = []



# Use the following code for KNN-only code execution

for i in range(num_data_splits):
    start_time = time.time()
    predictions = k_nearest_neighbors(train_sets[i], train_labels, test_sets[i], k=3, distance_measure=euclidean_distance)
    end_time = time.time()

    # Calculate accuracy for the current data split
    accuracy = np.mean(predictions == test_labels)
    accuracy_scores.append(accuracy)
    computation_time = end_time - start_time
    computation_times.append(computation_time)

# Calculate average accuracy and standard deviation
average_accuracy = np.mean(accuracy_scores)
standard_deviation = np.std(accuracy_scores)


# Unless mentioned, assume k = 3, train_imgs = 150, test_imgs = 20, subjects = 10, distance measure = euclidean
print("Results without applying PCA and using Euclidean as the distance measure in KNN (k = 3, train_images = 120, train_images = 50)\n")
print("Average Accuracy:", average_accuracy)
print("Standard Deviation:", standard_deviation)
print("Computation Times:", computation_times)



# Use the following code for PCA+KNN code execution

num_principal_components = 60
accuracy_scores = []
computation_times = []
cov_matrices_before = []
cov_matrices_after = []

for i in range(num_data_splits):
    start_time = time.time()
    accuracy = evaluate_pca_knn(train_sets[i], train_labels, test_sets[i], test_labels, k=3, distance_measure=euclidean_distance, num_components=num_principal_components)
    end_time = time.time()

    accuracy_scores.append(accuracy)
    computation_time = end_time - start_time
    computation_times.append(computation_time)

    cov_matrix_before = np.cov(np.transpose(train_sets[i]))
    cov_matrices_before.append(cov_matrix_before)

    pca, X_train_transformed = perform_pca(train_sets[i], num_principal_components)
    cov_matrix_after = np.cov(np.transpose(X_train_transformed))
    cov_matrices_after.append(cov_matrix_after)


average_accuracy = np.mean(accuracy_scores)
standard_deviation = np.std(accuracy_scores)

# Unless mentioned, assume k = 3, train_imgs = 150, test_imgs = 20, subjects = 10, distance measure = euclidean, principal components = 20
print("Results PCA + KNN using Euclidean as the distance measure (k = 3, principal components = 60)\n")
print("Number of Principal Components:", num_principal_components)
print("Average Accuracy Scores:", accuracy_scores)
print("Computation Times:", computation_times)
print("Average Accuracy:", average_accuracy)
print("Standard Deviation:", standard_deviation)

# Visualize covariance matrices

for i in range(num_data_splits):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cov_matrices_before[i], cmap='hot', interpolation='nearest')
    plt.title("Covariance Matrix Before PCA")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cov_matrices_after[i], cmap='hot', interpolation='nearest')
    plt.title("Covariance Matrix After PCA")
    plt.axis('off')

    plt.tight_layout()
    plt.show()









