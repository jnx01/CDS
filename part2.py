import numpy as np
import cv2
from sklearn.cluster import KMeans


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def kmeans_clustering(k, datapoints, kmeans_type):
    max_iterations = 100

    if kmeans_type == 1:

        centroids = datapoints[np.random.choice(range(len(datapoints)), k, replace=False)]
        cluster_indices = [[] for _ in range(k)]

        for _ in range(max_iterations):

            # Assign clusters
            for i in range(len(datapoints)):
                point = datapoints[i]
                distances = [euclidean_distance(point, centroid) for centroid in centroids]
                cluster_idx = np.argmin(distances)
                cluster_indices[cluster_idx].append(point)

            # Update centroids
            for j in range(k):
                cluster_points = np.array(cluster_indices[j])
                centroids[j] = np.mean(cluster_points, axis=0)

    else:

        kmeans = KMeans(n_clusters=k, max_iter=max_iterations, random_state=0)
        kmeans.fit(datapoints)
        cluster_indices = kmeans.labels_
        centroids = kmeans.cluster_centers_

    return cluster_indices, centroids


def extract_seed_pixel_coordinates(auxiliary_image):
    # Convert to RGB color space
    auxiliary_image_rgb = cv2.cvtColor(auxiliary_image, cv2.COLOR_BGR2RGB)

    # Reshape the auxiliary image to 2D
    height, width, _ = auxiliary_image_rgb.shape
    auxiliary_image_2d = auxiliary_image_rgb.reshape(height * width, -1)

    # Extract the coordinates of red pixels as foreground seeds
    foreground_pixel_indices = np.where(
        (auxiliary_image_2d[:, 0] >= 200) & (auxiliary_image_2d[:, 1] <= 50) & (auxiliary_image_2d[:, 2] <= 50))
    foreground_seeds = foreground_pixel_indices[0]

    # Extract the coordinates of blue pixels as background seeds
    background_pixel_indices = np.where(
        (auxiliary_image_2d[:, 0] <= 50) & (auxiliary_image_2d[:, 1] <= 50) & (auxiliary_image_2d[:, 2] >= 200))
    background_seeds = background_pixel_indices[0]

    return foreground_seeds, background_seeds


def extract_seed_pixels(image, foreground_seeds, background_seeds):
    # Convert to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to 2D
    height, width, _ = image_rgb.shape
    image_2d = image_rgb.reshape(height * width, -1)

    # Extract the foreground pixels using the seed coordinates
    foreground_pixels = image_2d[foreground_seeds]

    # Extract the background pixels using the seed coordinates
    background_pixels = image_2d[background_seeds]

    return foreground_pixels, background_pixels


def cluster_likelihood(pixel, centroid):
    distance = euclidean_distance(pixel, centroid)
    return np.exp(-distance)


def class_likelihood(pixels, frg_centroids, bkg_centroids, frg_cluster_indices, bkg_cluster_indices,
                     len_foreground_pixels, len_background_pixels, kmeans_type):
    frg_class = []
    bkg_class = []

    frg_cluster_weights = np.zeros(len(frg_centroids))
    bkg_cluster_weights = np.zeros(len(bkg_centroids))

    # Calculate weight of each cluster
    if kmeans_type == 1:  # custom kmeans

        for i in range(len(frg_cluster_indices)):
            frg_cluster_weights[i] = len(frg_cluster_indices[i]) / len_foreground_pixels

        for j in range(len(bkg_cluster_indices)):
            bkg_cluster_weights[j] = len(bkg_cluster_indices[j]) / len_background_pixels

    else:

        frg_cluster_counts = np.bincount(frg_cluster_indices)
        frg_cluster_weights = frg_cluster_counts / len_foreground_pixels

        bkg_cluster_counts = np.bincount(bkg_cluster_indices)
        bkg_cluster_weights = bkg_cluster_counts / len_background_pixels

    for pixel in pixels:
        # Calculate likelihood of belonging to each cluster in class
        frg_cluster_likelihood = [cluster_likelihood(pixel, frg_centroids[i]) for i in range(len(frg_centroids))]
        bkg_cluster_likelihood = [cluster_likelihood(pixel, bkg_centroids[j]) for j in range(len(bkg_centroids))]

        # Calculate class likelihood
        frg_class_likelihood = np.sum(frg_cluster_likelihood * frg_cluster_weights)
        bkg_class_likelihood = np.sum(bkg_cluster_likelihood * bkg_cluster_weights)

        # Assign class
        if frg_class_likelihood >= bkg_class_likelihood:
            frg_class.append(pixel)
        else:
            bkg_class.append(pixel)

    return frg_class, bkg_class


N = 88
kmeans_type = 1

auxiliary_image = cv2.imread('/content/van Gogh stroke.png')
image = cv2.imread('/content/van Gogh.PNG')

foreground_seeds, background_seeds = extract_seed_pixel_coordinates(auxiliary_image)
foreground_pixels, background_pixels = extract_seed_pixels(image, foreground_seeds, background_seeds)

frg_cluster_indices, frg_centroids = kmeans_clustering(N, foreground_pixels, kmeans_type)
bkg_cluster_indices, bkg_centroids = kmeans_clustering(N, background_pixels, kmeans_type)

frg_class, bkg_class = class_likelihood(foreground_pixels, frg_centroids, bkg_centroids, frg_cluster_indices,
                                        bkg_cluster_indices, len(foreground_pixels), len(background_pixels),
                                        kmeans_type)
print('Class likelihood of foreground seed pixels, using custom KMeans algorithm (N={})'.format(N))

print("Total pixels labeled 'Foreground': ", len(frg_class))
print("Total pixels labeled 'Background': ", len(bkg_class))
print('Accuracy: ', ((len(frg_class) / len(foreground_pixels)) * 100))
print('Foreground class: ', frg_class)
print('Background class: ', bkg_class)

frg_class, bkg_class = class_likelihood(background_pixels, frg_centroids, bkg_centroids, frg_cluster_indices,
                                        bkg_cluster_indices, len(foreground_pixels), len(background_pixels),
                                        kmeans_type)
print('\n\nClass likelihood of background seed pixels, using custom KMeans algorithm (N={})'.format(N))
print("Total pixels labeled 'Foreground': ", len(frg_class))
print("Total pixels labeled 'Background': ", len(bkg_class))
print('Accuracy: ', ((len(bkg_class) / len(background_pixels)) * 100))
print('Foreground class: ', frg_class)
print('Background class: ', bkg_class)

