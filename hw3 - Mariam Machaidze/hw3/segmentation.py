import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        # each point assigned to the closest centers
        for i in range(N):
            distances = np.linalg.norm(features[i] - centers, axis=1)
            assignments[i] = np.argmin(distances)

        # new center of each cluster
        new_centers = np.zeros_like(centers)
        for j in range(k):
            cluster_points = []
            for a in range(N):
                if assignments[a] == j:
                    cluster_points.append(features[a])

            # If there are points assigned to this cluster, calculate the mean
            if len(cluster_points) > 0:
                cluster_points = np.array(cluster_points)  # Convert to a numpy array for mean calculation
                new_center = cluster_points.mean(axis=0)  # Calculate mean of points in the cluster
                new_centers[j] = new_center  # Update the center for cluster j
            else:
                # If no points are assigned, retain the old center
                new_centers[j] = centers[j]

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        # Compute the distance from each point to each center
        distances = np.linalg.norm(features[:, np.newaxis, :] - centers, axis=2)
        
        # Assign each point to the nearest center
        new_assignments = np.argmin(distances, axis=1)

        # Step 4: Stop if cluster assignments did not change
        if np.array_equal(new_assignments, assignments):
            break

        assignments = new_assignments

        # Step 3: Compute new center of each cluster
        for j in range(k):
            if np.any(assignments == j):
                centers[j] = features[assignments == j].mean(axis=0)

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    # Step 2: Compute initial distance matrix between all clusters
    distance_matrix = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(i + 1, N):
            distance_matrix[i, j] = np.linalg.norm(centers[i] - centers[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric matrix

    while n_clusters > k:
        # Find the closest pair of clusters
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

        # Step 4: Merge cluster j into cluster i
        assignments[assignments == j] = i  # Update all points in cluster j to be in cluster i
        cluster_points = features[assignments == i]  # Points in the new merged cluster
        centers[i] = cluster_points.mean(axis=0)  # Update the center of the merged cluster

        # Step 5: Update the distance matrix for the merged cluster
        for m in range(N):
            if m != i and assignments[m] != i:
                distance_matrix[i, m] = np.linalg.norm(centers[i] - centers[m])
                distance_matrix[m, i] = distance_matrix[i, m]

        distance_matrix[:, j] = np.inf  # Mark the old cluster as inactive
        distance_matrix[j, :] = np.inf  # Mark the old cluster as inactive

        n_clusters -= 1
    
    # Step 6: Reassign final cluster labels
    unique_clusters = np.unique(assignments)
    final_assignments = np.zeros(N, dtype=int)
    for idx, cluster in enumerate(unique_clusters):
        final_assignments[assignments == cluster] = idx

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = img.reshape(H * W, C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    pass

    color_features = color.reshape(H * W, C)
    
    # Create the position features using np.mgrid to get x, y coordinates
    x_coords, y_coords = np.mgrid[0:H, 0:W]
    x_coords = x_coords.reshape(-1, 1)  # Shape (H * W, 1)
    y_coords = y_coords.reshape(-1, 1)  # Shape (H * W, 1)
    
    # Stack color and position features together
    features = np.hstack((color_features, x_coords, y_coords))  # Shape (H * W, C+2)

    # Normalize the features (column-wise)
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    pass

    # Count the number of matching pixels (True positives + True negatives)
    correct_pixels = (mask_gt == mask).sum()
    
    # Calculate the total number of pixels
    total_pixels = mask_gt.size
    
    # Compute accuracy as the fraction of correct pixels
    accuracy = correct_pixels / total_pixels

    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
