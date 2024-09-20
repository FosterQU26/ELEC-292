import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

#PART 1

# Read the CSV file and drop the first column
dataset = pd.read_csv('winequality.csv').iloc[:, 1:]

# Convert quality column to binary classification
dataset.loc[dataset['quality'] <= 5, 'quality'] = 0
dataset.loc[dataset['quality'] >= 6, 'quality'] = 1

# Define standard scaler to normalize outputs
scaler = StandardScaler()

# Define logistic regression model
log_reg = LogisticRegression(max_iter=10000)

# Create a pipeline with standard scaler and logistic regression
clf = make_pipeline(StandardScaler(), log_reg)

# Assign labels and data
labels = dataset.iloc[:, -1]
data = dataset.iloc[:, :-1]

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

# Train and test the model
clf.fit(X_train, Y_train)

# Calculate predictions and probabilities
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)
print("Probability:", y_clf_prob)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate F1 score
f1 = f1_score(Y_test, y_pred)
print("F1 Score:", f1)

# Calculate AUC
roc_auc = roc_auc_score(Y_test, y_clf_prob[:, 1])
print("The AUC is:", roc_auc)

# Plot ROC curve
plt.figure()
fpr, tpr, _ = roc_curve(Y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# PART 2

scaler = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)
pca = PCA(n_components=2)


# Create PCA pipeline
pca_pipe = make_pipeline(
    scaler,  # Data normalization
    pca   # PCA
)

# Apply the PCA pipeline to X_train and X_test
X_train_pca = pca_pipe.fit_transform(X_train)
X_test_pca = pca_pipe.transform(X_test)


# Create a pipeline with logistic regression only
clf = make_pipeline(scaler, l_reg)

#Train 'clf' with X_train_pca and Y_train
clf.fit(X_train_pca, Y_train)

#Obtain predictions for X_test_pca
Y_pred_pca = clf.predict(X_test_pca)

#Create the decision boundary display
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train_pca, response_method="predict",
    xlabel='X1', ylabel='X2',
    alpha=0.5,
)

# Visualize training set samples along with decision boundary
disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.show()
accuracy = accuracy_score(Y_test, Y_pred_pca)
print("The accuracy is ", accuracy)
