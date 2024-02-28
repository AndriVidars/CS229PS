import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=None):
        # Initialize the Decision Tree with an optional max depth parameter.
        self.max_depth = max_depth
        
    def fit(self, X, y):
        # Fit the decision tree model to the training data.
        self.n_classes_ = len(np.unique(y))  # Calculate the number of unique classes.
        self.n_features_ = X.shape[1]  # Number of features.
        self.tree_ = self._grow_tree(X, y)  # Build the decision tree. Return the root node.
        
    def predict(self, X):
        # Predict class labels for samples in X.
        return [self._predict(inputs) for inputs in X]
        
    def _misclassification_loss(self, y):
        loss = None
        # *** START CODE HERE ***
        if y.shape[0] == 0:
            return 1

        c = Counter(y.tolist())
        loss = 1 - c.most_common(1)[0][1]/y.shape[0]

        # *** END YOUR CODE ***
        # Return the misclassification loss.
        return loss
        
    def _best_split(self, X, y):
        # TODO: Find the best split for a node.
        # Hint: Iterate through all features and calculate the best split threshold based on the misclassification loss.
        # Hint: You might want to loop through all the unique values in the feature to find the best threshold.
        best_idx, best_thr = None, None
        # *** START CODE HERE ***        
        # Calculate the parent's loss to compare with splits
        parent_loss = self._misclassification_loss(y)
        # if parent_loss is 0, then don't enter loop, cannot improve tree further
        if parent_loss == 0:
            return best_idx, best_thr

        min_loss = float('inf')
        for i in range(self.n_features_):
            # find threshold
            features_sorted = sorted([X[j][i] for j in range(X.shape[0])])
            thr_vals = [(features_sorted[i] + features_sorted[i+1])/2 for i in range(X.shape[0] - 1)]
            for thr in thr_vals:
                y_right = y[X[:, i] >= thr]
                y_left = y[X[:, i] < thr]
                split_loss = ((y_left.shape[0]/y.shape[0]) * self._misclassification_loss(y_left) +
                                             (y_right.shape[0]/y.shape[0]) * self._misclassification_loss(y_right))
                
                if split_loss < min_loss:
                    min_loss = split_loss
                    best_idx, best_thr = i, thr
    
        # *** END YOUR CODE ***
        # Return the best split with the feature index and threshold.
        return best_idx, best_thr
        
    def _grow_tree(self, X, y, depth=0):
        # Build a decision tree by recursively finding the best split.
        # param depth is the current depth of the tree.
        # Construct the root node.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        root_node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            # TODO: Find the best split using _best_split and grow the tree recursively.
            # *** START YOUR CODE ***
            split_idx, thr = self._best_split(X, y)
            if thr == None:
                # cannot improve tree further
                return root_node

            root_node.feature_index = split_idx
            root_node.threshold = thr

            y_left, X_left = y[X[:, split_idx] < thr], X[X[:, split_idx] < thr]
            y_right, X_right = y[X[:, split_idx] >= thr], X[X[:, split_idx] >= thr]
            root_node.left = self._grow_tree(X_left, y_left, depth + 1)
            root_node.right = self._grow_tree(X_right, y_right, depth + 1)

            # *** END YOUR CODE ***
        # Return the root node.
        return root_node
        
    def _predict(self, inputs):
        # Predict the class of ONE input based on the tree structure.
        node = self.tree_
        while node.left:
            # TODO: Traverse the tree to find the corresponding node and predict the class of the input.
            # Hint: iteratively update the node to be its left or right child until a leaf node is reached.
            # *** START YOUR CODE ***
            idx, thr = node.feature_index, node.threshold
            if inputs[idx] < thr:
                node = node.left
            else:
                node = node.right
            # *** END YOUR CODE ***
        return node.predicted_class
    
class Node:
    def __init__(self, *, predicted_class):
        self.predicted_class = predicted_class  # Class predicted by this node
        self.feature_index = None  # Index of the feature used for splitting
        self.threshold = None  # Threshold value for splitting
        self.left = None  # Left child
        self.right = None  # Right child

    def is_leaf_node(self):
        # Check if this node is a leaf node.
        return self.left is None and self.right is None


if __name__ == "__main__":
    data = pd.DataFrame({
        "Age": [24, 53, 23, 25, 32, 52, 22, 43, 52, 48], 
        "Salary": [40, 52, 25, 77, 48, 110, 38, 44, 27, 65], 
        "College": [1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    })
    # Create a simple dataset for testing the decision tree
    X = np.array(data[["Age", "Salary"]])
    y = np.array(data["College"])
    
    # Initialize and fit the decision tree
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)
    
    # Print the classification accuracy
    print(f"Accuracy for college degree dataset: {round(np.mean(clf.predict(X) == y)*100, 2)}%")
    
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    # print("Multi-class labels", np.unique(y))
    
    # Split the data into training and testing sets
    # DO NOT MODIFY THE RANDOM STATE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=229)

    # Train the decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = tree.predict(X_test)

    # Compute the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy for iris dataset: {round(accuracy*100, 2)}%")
