import numpy as np
from collections import Counter


# Global Function
def euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    
    def __init__(self, k = 3):  # default k value 3
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    
    def predict(self, X):
        predicted_labels = [(self._predict(x)) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        #compute the distance from the X_train datapoints
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train ]
        #sort the distances top k samples
        k_indices = np.argsort(distances)[:self.k]
        #k-nearest labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #majority vote to get common class labels
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]
    
    
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    def accuracy(true, pred):
        return np.sum(true == pred) / len(y_test)
    
    iris = datasets.load_iris()
    X , y = iris.data , iris.target
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1234 )
    
    for k in range(3,20):
        clf = KNN(k=k)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print(f"KNN classification accuracy  for k value: {k} is", accuracy(y_test, predictions))
    