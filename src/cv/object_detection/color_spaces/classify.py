from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from ..feature_extraction.main import normalize_data

car_features = []
notcar_features = []

y = np.hstack((np.ones(len(car_features)),
              np.zeros(len(notcar_features))))

X = np.vstack((car_features, notcar_features)).astype(np.float64)

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

scaled_X_train = normalize_data(X_train)
scaled_X_test = normalize_data(X_test)

svc = LinearSVC()
svc.fit(scaled_X_train, y_train)

