from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    ################################################################################
    # TODO:                                                                        #
    # Implement the function to compute the distance between each test point       #
    # in X and each training point in self.X_train using a nested loop.            #
    #                                                                              #
    # You should compute the L2 distance between the ith test point and the jth    #
    # training point, and store the result in dists[i, j].                         #
    #                                                                              #
    # You should not use np.linalg.norm() or a loop over the dimensions (D).       #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                # Compute the Euclidean distance between the ith test point and the jth training point.
                distance = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
                dists[i, j] = distance

        return dists
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    ################################################################################
    # TODO:                                                                        #
    # Implement the function to compute the distance between each test point       #
    # in X and each training point in self.X_train using a single loop over        #
    # the test data.                                                               #
    #                                                                              #
    # You should compute the L2 distance between the ith test point and all        #
    # training points, and store the result in dists[i, :].                        #
    # Do not use np.linalg.norm().                                                 #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            # Vectorized computation of distances between the ith test example and all training examples.
            # Using broadcasting and the formula: sqrt((x - y)^2) = sqrt(x^2 + y^2 - 2xy)
            dists[i, :] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
        
        return dists

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ################################################################################
    # TODO:                                                                        #
    # Implement the function to compute the distance between each test point       #
    # in X and each training point in self.X_train using no explicit loops.        #
    #                                                                              #
    # You should compute the L2 distance between all test points and all           #
    # training points without using any explicit loops. Store the result in dists. #
    #                                                                              #
    # You should implement this function using only basic array operations;        #
    # do not use scipy functions or np.linalg.norm().                              #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        # Get the number of test and training points
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # Step 1: Compute the squared sum of each test point (X^2), shape: (num_test, 1)
        X_square = np.sum(X**2, axis=1).reshape(num_test, 1)

        # Step 2: Compute the squared sum of each training point (Y^2), shape: (num_train,)
        X_train_square = np.sum(self.X_train**2, axis=1)

        # Step 3: Compute the cross term (X * Y), shape: (num_test, num_train)
        cross_term = np.dot(X, self.X_train.T)

        # Step 4: Compute the distance matrix using the formula: dist(x, y) = sqrt(x^2 + y^2 - 2xy)
        dists = np.sqrt(X_square + X_train_square - 2 * cross_term)


        return dists

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            sorted_indices = np.argsort(dists[i])  # Get indices that would sort the distances
            closest_y = self.y_train[sorted_indices[:k]]  # Get the labels of the k nearest neighbors

            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            y_pred[i] = np.bincount(closest_y).argmax()
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
