# KNN Extended Project — Progress and Continuation Notes

## 1. Project Goal

This project is an extension of the user's earlier Coursera guided project:

> **Simple Nearest Neighbors Regression and Classification**
>
> Completed: August 12, 2023  
> Grade: 83.33%

The original course covered:
- Basic KNN principles
- KNN classification
- KNN regression
- Making predictions
- Small hand-worked KNN examples
- Python implementation

The new project is being rebuilt from scratch and extended to demonstrate a more complete machine-learning workflow.

### Main extension

The project will specifically investigate:

1. **Feature Scaling**
   - No scaling
   - Standardization
   - Min-Max scaling

2. **Hyperparameter Tuning**
   - Number of neighbors K
   - Distance metric
   - Potentially distance weighting

3. **Cross-validation**
4. **GridSearchCV**
5. **Pipeline-based preprocessing and modeling**
6. **Proper evaluation and comparison**
7. **Understanding data leakage**

The final project should demonstrate that we understand not just how to call a KNN implementation, but why KNN behaves the way it does and how to optimize it correctly.

---

# 2. Planned Project Structure

Current recommended structure:

```text
KNN_Extended_Project/
│
├── 01_KNN_From_Scratch.ipynb
│
├── 02_Feature_Scaling_Experiment.ipynb
│
└── README.md
```

### `01_KNN_From_Scratch.ipynb`

This notebook contains:
- KNN intuition
- Euclidean distance
- Manhattan distance
- KNN classification from scratch
- Majority voting
- Effect of K
- Manual accuracy
- Comparison with Scikit-learn

### `02_Feature_Scaling_Experiment.ipynb`

This will contain the actual ML experiment:
- Dataset loading
- Dataset inspection
- Feature/target selection
- Train/test split
- Baseline KNN
- No scaling
- StandardScaler
- MinMaxScaler
- Performance comparison
- Hyperparameter tuning
- Cross-validation
- GridSearchCV
- Pipeline
- Final evaluation

### Dataset selected

The user has selected the Kaggle:

**Customer Support Ticket Dataset**

Kaggle URL:

https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

IMPORTANT:
- The dataset has been selected.
- We have NOT yet inspected its actual columns in this project.
- The next chat should begin by loading/inspecting the dataset and deciding what meaningful prediction target can be used for KNN.
- Do not assume the target column or feature types before inspecting the actual file.

---

# 3. KNN Fundamentals

## 3.1 What is KNN?

K-Nearest Neighbors is a supervised machine-learning algorithm based on the idea that nearby observations tend to have similar outcomes.

Given:
- Training observations
- A new/unknown observation
- A distance metric
- A number K

KNN finds the K training observations closest to the new observation.

For classification, the neighbors vote.

For regression, their target values are generally averaged.

---

# 4. KNN Classification

Suppose training data contains:

| Height | Weight | Class |
|---:|---:|---|
| 160 | 55 | A |
| 165 | 60 | A |
| 170 | 65 | A |
| 180 | 80 | B |
| 185 | 85 | B |
| 190 | 90 | B |

For a new observation:

x = (175,72)

KNN calculates distances from this point to all training points.

If the three nearest classes are:

```text
A
B
B
```

then majority voting gives:

prediction = B

---

# 5. Euclidean Distance

For two points:

x = (x1, x2)

and:

y = (y1, y2)

Euclidean distance is:

d_E(x,y) = sqrt((x1-y1)^2 + (x2-y2)^2)

More generally:

d_E(x,y) = sqrt(sum_j (x_j-y_j)^2)

where p is the number of features.

---

# 6. Manhattan Distance

Manhattan distance is:

d_M(x,y) = sum_j |x_j-y_j|

For two dimensions:

d_M(x,y) = |x1-y1| + |x2-y2|

Example:

A = (2,3)

B = (7,9)

Euclidean:

d_E = sqrt((7-2)^2 + (9-3)^2)
    = sqrt(61)
    ≈ 7.81

Manhattan:

d_M = |7-2| + |9-3|
    = 11

---

# 7. Basic Distance Functions Implemented

```python
import numpy as np


def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two points.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """
    Calculate the Manhattan distance between two points.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    return np.sum(np.abs(x1 - x2))
```

Example:

```python
x1 = np.array([1, 1])
x2 = np.array([4, 5])

print("Euclidean:", euclidean_distance(x1, x2))
print("Manhattan:", manhattan_distance(x1, x2))
```

Expected:

```text
Euclidean: 5.0
Manhattan: 7
```

---

# 8. KNN Classification Process

For classification:

```text
Given:
    Training data
    New point x
    Number of neighbors K

1. Calculate distance from x to every training point.

2. Sort points by distance.

3. Select the K closest points.

4. Retrieve their class labels.

5. Perform majority voting.

6. Return the majority class.
```

---

# 9. Small Classification Dataset

We created:

```python
X_train = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [5, 5],
    [6, 5],
    [5, 6]
])

y_train = np.array([
    "A",
    "A",
    "A",
    "B",
    "B",
    "B"
])
```

For:

```python
x = np.array([2, 2])
```

the Euclidean distances are approximately:

```text
[1.414, 1.000, 1.000, 4.243, 5.000, 5.000]
```

The three nearest points are indices:

```text
[1, 2, 0]
```

Their labels are:

```text
['A', 'A', 'A']
```

Therefore:

```text
prediction = A
```

---

# 10. Calculate Distances to All Training Points

Implemented:

```python
def calculate_distances(X_train, x, metric="euclidean"):
    """
    Calculate the distance between x and every point in X_train.
    """

    distances = []

    for point in X_train:

        if metric == "euclidean":
            distance = euclidean_distance(point, x)

        elif metric == "manhattan":
            distance = manhattan_distance(point, x)

        else:
            raise ValueError(
                "metric must be 'euclidean' or 'manhattan'"
            )

        distances.append(distance)

    return np.array(distances)
```

---

# 11. Majority Voting

Implemented:

```python
def majority_vote(labels):
    """
    Return the most common label.
    """

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    return unique_labels[np.argmax(counts)]
```

Example:

```python
labels = np.array(["A", "B", "A", "A", "B"])

print(majority_vote(labels))
```

Output:

```text
A
```

because:

A = 3, B = 2

---

# 12. Complete KNN Classifier From Scratch

Final version developed so far:

```python
class KNNClassifier:

    def __init__(self, k=3, metric="euclidean"):

        if k <= 0:
            raise ValueError("k must be greater than 0")

        if metric not in ["euclidean", "manhattan"]:
            raise ValueError(
                "metric must be 'euclidean' or 'manhattan'"
            )

        self.k = k
        self.metric = metric

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples"
            )

        if self.k > len(X):
            raise ValueError(
                "k cannot be larger than the number of training samples"
            )

        self.X_train = X
        self.y_train = y

        return self

    def predict_one(self, x):

        distances = calculate_distances(
            self.X_train,
            x,
            metric=self.metric
        )

        nearest_indices = np.argsort(distances)[:self.k]

        neighbor_labels = self.y_train[nearest_indices]

        return majority_vote(neighbor_labels)

    def predict(self, X):

        X = np.asarray(X)

        predictions = [
            self.predict_one(x)
            for x in X
        ]

        return np.asarray(predictions)
```

---

# 13. Testing the From-Scratch Classifier

Example:

```python
knn = KNNClassifier(
    k=3,
    metric="euclidean"
)

knn.fit(X_train, y_train)

X_test = np.array([
    [2, 2],
    [5, 5],
    [3, 3],
    [4, 4]
])

predictions = knn.predict(X_test)

print(predictions)
```

The predictions obtained were:

```text
['A' 'B' 'A' 'B']
```

---

# 14. KNN Regression

KNN can also solve regression problems.

Suppose the nearest neighbors have target values:

230, 300, 360

A basic KNN regression prediction is:

y_hat = (230 + 300 + 360) / 3
      = 296.67

Thus:

### Classification

majority vote

### Regression

average of neighbor target values

A regression implementation has not yet been built in this project. It can be added later if useful, but the current extension is primarily focused on classification and feature scaling/hyperparameter optimization.

---

# 15. Effect of K

K is a hyperparameter.

Small K:

```text
Small K
   ↓
Very local decisions
   ↓
Sensitive to individual observations/noise
   ↓
Potentially high variance
```

Large K:

```text
Large K
   ↓
Larger neighborhood
   ↓
Local structure becomes less important
   ↓
Potentially high bias
```

This creates a bias-variance tradeoff.

The important question is:

How should we choose K?

We will answer this experimentally using cross-validation and GridSearchCV.

---

# 16. Manual Accuracy

Implemented:

```python
def accuracy_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)
```

Accuracy:

accuracy = correct predictions / total predictions

Example:

```python
y_true = np.array(["A", "A", "B", "B", "B"])

y_pred = np.array(["A", "B", "B", "B", "B"])

accuracy = accuracy_score_manual(
    y_true,
    y_pred
)

print(accuracy)
```

Result:

```text
0.8
```

or:

80%

---

# 17. Comparison With Scikit-learn

We then compared our implementation with:

```python
from sklearn.neighbors import KNeighborsClassifier
```

Scikit-learn:

```python
sklearn_knn = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)

sklearn_knn.fit(
    X_train,
    y_train
)

sklearn_predictions = sklearn_knn.predict(X_test)
```

Our implementation and Scikit-learn produced the same predictions:

```text
Our implementation:
['A' 'B' 'A' 'B']

Scikit-learn:
['A' 'B' 'A' 'B']
```

This confirms that our basic implementation behaves consistently with Scikit-learn for this test case.

---

# 18. Why Implement From Scratch?

The project should not simply be:

```python
KNeighborsClassifier(...)
```

The from-scratch implementation demonstrates understanding of:

- Distance calculation
- Neighbor selection
- Sorting
- Majority voting
- The role of K
- Distance metrics

Then Scikit-learn will be used for the actual experimental workflow because it provides robust implementations and tools such as:

- KNeighborsClassifier
- StandardScaler
- MinMaxScaler
- Pipeline
- GridSearchCV
- Cross-validation utilities

Strong interview explanation:

> I first implemented KNN from scratch to understand the distance computation, neighbor selection, and majority-voting mechanism. I then used Scikit-learn for the experimental pipeline, particularly preprocessing, cross-validation, and hyperparameter optimization.

---

# 19. Feature Scaling

This is the main extension of the project.

KNN is distance-based.

For Euclidean distance:

d(x,y) = sqrt(sum_j (x_j-y_j)^2)

Therefore feature scale directly affects the distance.

Consider:

```text
Age:       18 – 80
Income:    20,000 – 2,000,000
```

Income has a much larger numerical magnitude.

For example:

Delta Age = 10

Delta Income = 100000

Then:

d = sqrt(10^2 + 100000^2)
  ≈ 100000

Thus income can dominate the distance even if age is also predictive.

This is a scale problem, not necessarily evidence that income is intrinsically more informative.

---

# 20. Important Observation About Units

Suppose income is represented as:

50000

instead of:

0.5

if the unit is changed from rupees to lakhs.

The underlying information has not changed, but the numerical distances have changed.

A distance-based algorithm can therefore behave differently simply because of the units used to represent a feature.

This motivates feature scaling.

---

# 21. Standardization

Standardization uses:

z = (x - mu) / sigma

where:
- x = original value
- mu = feature mean
- sigma = feature standard deviation

After standardization, the feature has approximately:

mean = 0

and:

standard deviation = 1

Manual implementation:

```python
def standardize(X):
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_scaled = (X - mean) / std

    return X_scaled
```

---

# 22. Min-Max Scaling

Min-Max scaling is:

x' = (x - x_min) / (x_max - x_min)

For the usual configuration, values are mapped to:

[0, 1]

Manual implementation:

```python
def minmax_scale(X):
    X = np.asarray(X, dtype=float)

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    X_scaled = (
        (X - minimum)
        / (maximum - minimum)
    )

    return X_scaled
```

---

# 23. Standardization vs. Min-Max Scaling

| Method | Formula | Main result |
|---|---|---|
| Standardization | (x - mu) / sigma | Mean approximately 0, std approximately 1 |
| Min-Max | (x - x_min) / (x_max - x_min) | Usually maps values to [0, 1] |

Neither method should automatically be assumed to be superior.

The project will compare them experimentally.

---

# 24. Scikit-learn Scaling

For real experiments:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

Standardization:

```python
scaler = StandardScaler()

X_standardized = scaler.fit_transform(X)
```

Min-Max:

```python
scaler = MinMaxScaler()

X_minmax = scaler.fit_transform(X)
```

---

# 25. Critical Point: fit vs transform

Suppose:

```python
X_train
X_test
```

The scaler must learn its parameters from the training set:

```python
scaler.fit(X_train)
```

Then:

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Do NOT independently fit the scaler on the test set:

```python
# WRONG
scaler.fit_transform(X_test)
```

Why?

Because test-set information would influence preprocessing.

This is a form of data leakage.

---

# 26. Why Pipeline Will Matter

Later we will use:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])
```

This becomes particularly important when using:

```python
GridSearchCV
```

because cross-validation must fit preprocessing only on the corresponding training folds.

A Pipeline helps ensure the preprocessing and model are handled correctly together.

---

# 27. Manual Scaling Experiment

We created a toy dataset:

```python
X = np.array([
    [20, 80000],
    [25, 60000],
    [30, 70000],
    [35, 40000],
    [40, 50000],
    [45, 30000]
])

y = np.array([
    "A",
    "A",
    "A",
    "B",
    "B",
    "B"
])
```

New observation:

```python
x_new = np.array([32, 55000])
```

We compared distances before and after scaling.

The purpose was to demonstrate that the ordering of nearest neighbors can change after scaling.

This toy experiment was for understanding only.

---

# 28. Actual Dataset for the Project

The user has chosen:

**Customer Support Ticket Dataset**

Kaggle:

https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

IMPORTANT:
We have not yet inspected the downloaded dataset in this project.

The next step must therefore be:

1. Obtain/load the dataset.
2. Inspect filenames.
3. Inspect shape.
4. Display columns.
5. Check data types.
6. Check missing values.
7. Check duplicate rows.
8. Inspect categorical and numerical variables.
9. Determine possible target variables.
10. Decide what KNN classification problem is meaningful.
11. Decide which features can reasonably be used for KNN.
12. Avoid blindly encoding arbitrary text columns.
13. Establish a reproducible train/test split.

Do NOT assume the target column before inspecting the actual dataset.

---

# 29. Next Notebook

Create:

```text
02_Feature_Scaling_Experiment.ipynb
```

The actual experimental pipeline should eventually look like:

```text
Customer Support Ticket Dataset
            |
            v
      Data inspection
            |
            v
    Define X and target y
            |
            v
      Train/test split
            |
            +----------------------+
            |                      |
            v                      v
       Baseline KNN          Preprocessed KNN
       (no scaling)          |
                              +-- StandardScaler
                              |
                              +-- MinMaxScaler
                                    |
                                    v
                                  KNN
                                    |
                                    v
                            Performance comparison
                                    |
                                    v
                             Hyperparameter tuning
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
                         K                 Distance metric
                         |                     |
                         +----------+----------+
                                    |
                                    v
                              GridSearchCV
                                    |
                                    v
                                 Pipeline
                                    |
                                    v
                              Final evaluation
```

---

# 30. Planned Hyperparameters

Main hyperparameters:

### Number of neighbors

K

For example:

```python
n_neighbors = [1, 3, 5, 7, 9, 11, 15, 21]
```

The exact range should be chosen after inspecting the dataset size.

### Distance metric

At minimum:

```text
euclidean
manhattan
```

Controlled through:

```python
metric
```

### Weighting

Potentially:

```text
uniform
distance
```

where:
- `uniform`: every neighbor has equal voting weight
- `distance`: closer neighbors receive greater influence

We should add this if it remains useful after the basic experiment.

---

# 31. Planned Evaluation

For classification, do not rely exclusively on accuracy.

Depending on target distribution, consider:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

If the target is imbalanced, accuracy may be misleading.

We should inspect class distribution before deciding which metric is most important.

---

# 32. Important Experimental Principle

When comparing scaling methods, keep other things controlled.

For example:

```text
Same dataset
Same train/test split
Same K
Same distance metric
Same evaluation metric
```

and change only:

```text
No scaling
vs
StandardScaler
vs
MinMaxScaler
```

Then we can make a defensible statement about the effect of scaling.

Later, when tuning hyperparameters, the experimental design changes intentionally.

---

# 33. Final Intended Project Story

### Stage 1 — Understand KNN

Implement the algorithm from scratch.

### Stage 2 — Establish baseline

Use Scikit-learn KNN on the selected dataset.

### Stage 3 — Identify a problem

Show that feature magnitude can distort distance calculations.

### Stage 4 — Apply scaling

Compare:
- no scaling
- standardization
- Min-Max scaling

### Stage 5 — Optimize KNN

Use cross-validation to search over:
- K
- distance metric
- potentially weighting

### Stage 6 — Prevent leakage

Use:

```python
Pipeline
```

with:

```python
GridSearchCV
```

### Stage 7 — Evaluate

Compare the final tuned model with the baseline.

### Stage 8 — Explain

Analyze:
- Why scaling helped or did not help
- Why the chosen K was selected
- Why the chosen distance metric performed better
- Whether the result generalizes
- Limitations of applying KNN to the dataset

---

# 34. Resume Direction

Do not yet claim the full project on the resume because the experimental portion has not been completed.

Potential final entry after completion:

**K-Nearest Neighbors Classification & Hyperparameter Optimization | Python, NumPy, Pandas, Scikit-learn**

- Reimplemented KNN classification from scratch using Euclidean and Manhattan distance metrics and majority-voting based prediction.
- Investigated the sensitivity of distance-based classification to feature magnitude by comparing unscaled, standardized, and Min-Max-scaled features.
- Applied cross-validation and GridSearchCV to optimize neighborhood size and distance metrics, comparing tuned KNN against a baseline model.
- Used preprocessing pipelines to prevent data leakage and evaluated model performance using classification metrics and confusion matrices.

This wording should only be finalized after the corresponding experiments are actually completed.

---

# 35. Current Status

### Completed

- [x] KNN concept
- [x] KNN classification intuition
- [x] Euclidean distance
- [x] Manhattan distance
- [x] Manual distance calculations
- [x] Neighbor selection
- [x] Majority voting
- [x] KNN classifier from scratch
- [x] Input validation in classifier
- [x] Manual accuracy
- [x] Comparison against Scikit-learn
- [x] Verified matching predictions:
  - Our implementation: `['A', 'B', 'A', 'B']`
  - Scikit-learn: `['A', 'B', 'A', 'B']`
- [x] K hyperparameter intuition
- [x] Feature scaling motivation
- [x] Standardization theory
- [x] Min-Max scaling theory
- [x] Manual scaling functions
- [x] Scikit-learn scaler usage
- [x] fit vs transform
- [x] Data leakage concept
- [x] Pipeline motivation

### Not yet completed

- [ ] Load actual Customer Support Ticket Dataset
- [ ] Inspect dataset
- [ ] Decide target variable
- [ ] Decide usable features
- [ ] Handle categorical variables
- [ ] Handle missing values
- [ ] Handle text/date columns if relevant
- [ ] Train/test split
- [ ] Baseline KNN
- [ ] Scaling experiment
- [ ] Performance comparison
- [ ] Cross-validation
- [ ] GridSearchCV
- [ ] Hyperparameter optimization
- [ ] Pipeline
- [ ] Final model
- [ ] Final analysis
- [ ] README
- [ ] Final resume bullets

---

# 36. Exact Starting Point for the Next Chat

Start the next chat with:

> Continue the KNN Extended Project from `KNN_Extended_Project_Progress.md`. We have finished the from-scratch KNN implementation and the feature-scaling theory. I want to start the actual experiment using the Kaggle Customer Support Ticket Dataset: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset. First inspect the actual dataset and determine a meaningful KNN classification target and feature set. Do not assume the columns or target before inspecting the data.

Then upload/provide the downloaded dataset if it is not already available in the new chat/project.

The first task should be **dataset inspection**, not modeling.
