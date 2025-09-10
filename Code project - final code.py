### Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np


### Importing the dataset
df = pd.read_excel("C:/Users/saram/OneDrive/Desktop/Uni/Triennale/Terzo anno/Secondo semestre/Machine Learning ad Artificial Intelligence/Code Project/data_academic_performance.xlsx")


### PROFILING
## Focus on the dataset
# Number of rows and columns
n_rows, n_columns = df.shape
print(f'Number of rows: {n_rows}, Number of columns: {n_columns}')

# Type of each column
column_types = df.dtypes
print('Type of each column:\n', column_types)

# Missing values
missing_values = df.isnull().sum()
print('Missing values:\n', missing_values)
df = df.drop(columns=['Unnamed: 9'], errors='ignore')

# Descriptive statistics
print('Descriptive statistics:\n', df.describe())

desc = df.describe()
selected_cols = ["MAT_S11", "CR_S11", "BIO_S11", "ENG_S11", "G_SC", "PERCENTILE", "2ND_DECILE", "QUARTILE"]
desc_subset = desc[selected_cols]
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=desc_subset.round(2).values,
                 rowLabels=desc_subset.index,
                 colLabels=desc_subset.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.1, 1.1)
for key, cell in table.get_celld().items():
    cell.set_height(0.07)
for col in range(len(desc_subset.columns)):
    cell = table[0, col]
    cell.get_text().set_weight('bold')
    cell.set_facecolor('#f0f0f0')
for row in range(1, len(desc_subset.index)+1):
    cell = table[row, -1]  # colonna con rowLabels
    cell.get_text().set_weight('bold')
    cell.set_facecolor('#f0f0f0')
plt.tight_layout()
plt.savefig("descriptive_statistics_table_elegant.png", dpi=300)
plt.show()

# Histograms describing numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns
for column in numerical_columns:
    plt.figure(figsize = (8, 4))
    sns.histplot(df[column], kde = True, bins = 30)
    plt.title(f'Graphical representation of {column}')
    plt.show()


## Focus on the variables
# Variable selection
selected_features = [
    'GENDER', 'EDU_FATHER', 'EDU_MOTHER', 'STRATUM', 'INTERNET',
    'COMPUTER', 'MOBILE', 'SCHOOL_NAT'
]
for col in ['MOBILE', 'SCHOOL_NAT', 'INTERNET', 'COMPUTER', 'GENDER']:
    print(f"Distribution of {col}:\n{df[col].value_counts()}\n")

# Defining the variables
ordinal_cols = ['STRATUM', 'EDU_FATHER', 'EDU_MOTHER']
nominal_cols = ['MOBILE', 'SCHOOL_NAT', 'INTERNET', 'COMPUTER', 'GENDER']
for col in ['EDU_FATHER', 'EDU_MOTHER']:
    df = df[~((df[col] == 0) | (df[col] == 'Not sure') | (df[col] == 'Ninguno'))]
df = df[~(df['STRATUM'] == 0)]
for col in ['EDU_FATHER', 'EDU_MOTHER', 'STRATUM']:
    df[col] = df[col].astype(str).str.strip()

# Ordinal encoding
edu_order = [
    'Incomplete primary', 
    'Complete primary', 
    'Incomplete Secundary', 
    'Incomplete technical or technological', 
    'Incomplete Professional Education', 
    'Complete Secundary',
    'Complete technique or technology',
    'Complete professional education',
    'Postgraduate education'
]

stratum_order = ['Stratum 1', 'Stratum 2', 'Stratum 3', 'Stratum 4', 'Stratum 5', 'Stratum 6']

# Creating the ordinal encoder
ordinal_encoder = OrdinalEncoder(categories=[stratum_order, edu_order, edu_order])

# Creating the nominal encoder
nominal_encoder = OneHotEncoder(drop='first', sparse_output=False)

# Combining the two encoders into a single preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('ord', ordinal_encoder, ordinal_cols),
    ('nom', nominal_encoder, nominal_cols)
])

# Defining the target variable
df['PERFORMANCE_CLASS'] = pd.qcut(df['PERCENTILE'], q=3, labels=['Low', 'Medium', 'High'])
target = 'PERFORMANCE_CLASS'
X = df[selected_features]
y = df[target]

# Representing graphically the target variable
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='PERFORMANCE_CLASS', order=['Low', 'Medium', 'High'], palette='Blues')
plt.title('Distribution of the target variable', fontsize=14, fontweight='bold')
plt.xlabel('Performance class')
plt.ylabel('Number of observations')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Representing graphically some of the main features
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
blues = get_cmap("Blues")
fig = plt.figure(figsize=(16, 40))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1.2, 1], hspace=0.6)
# - Gender
ax1 = fig.add_subplot(gs[0, 0])
df['GENDER'].value_counts().plot(kind='bar', ax=ax1, color=blues(0.5))
ax1.set_title('(a) Gender')
ax1.set_ylabel('Count')
ax1.set_xlabel('')
ax1.set_xticklabels(df['GENDER'].value_counts().index, rotation=0, ha='center')
# - School Nature
ax2 = fig.add_subplot(gs[0, 1])
df['SCHOOL_NAT'].value_counts().plot(kind='bar', ax=ax2, color=blues(0.6))
ax2.set_title('(b) School Nature')
ax2.set_ylabel('Count')
ax2.set_xlabel('')
ax2.set_xticklabels(df['SCHOOL_NAT'].value_counts().index, rotation=0, ha='center')
# - Mobile, Internet, Computer
ax3 = fig.add_subplot(gs[1, 0])
binary_vars = ['MOBILE', 'INTERNET', 'COMPUTER']
base_colors = [0.55, 0.65, 0.75]
xticks_labels = []
xticks_positions = []
position = 0
bar_width = 0.35
for i, var in enumerate(binary_vars):
    counts = df[var].value_counts().sort_index()
    yes_color_val = max(base_colors[i] - 0.15, 0)
    yes_color = blues(yes_color_val)
    no_color = blues(base_colors[i])
    ax3.bar(position, counts.get(1, 0), width=bar_width, color=yes_color)
    ax3.bar(position + bar_width, counts.get(0, 0), width=bar_width, color=no_color)
    xticks_positions.extend([position + bar_width/2, position + 1.5*bar_width])
    xticks_labels.extend([f'{var}_Yes', f'{var}_No'])
    position += 2
ax3.set_title('(c) Access to Devices')
ax3.set_ylabel('Count')
ax3.set_xticks(xticks_positions)
ax3.set_xticklabels(xticks_labels, rotation=45, ha='right')
for tick in ax3.get_xticklabels():
    tick.set_y(0.07)
# - Edu Mother & Edu Father
ax4 = fig.add_subplot(gs[1, 1])
edu_mother_counts = df['EDU_MOTHER'].value_counts().sort_index()
edu_father_counts = df['EDU_FATHER'].value_counts().sort_index()
x = np.arange(len(edu_mother_counts.index))
width = 0.35
ax4.bar(x - width/2, edu_mother_counts.values, width, label='Mother', color=blues(0.4))
ax4.bar(x + width/2, edu_father_counts.values, width, label='Father', color=blues(0.8))
ax4.set_xticks(x)
custom_labels = ['Compl. Secundary', 'Compl. Primary', 'Compl. Professional', 'Compl. technical', 'Incompl. Professional', 'Incompl. Secundary', 'Incompl. Primary', 'Incompl. technical', 'Postgraduate']
ax4.set_xticklabels(custom_labels, rotation=45, ha='right', fontsize=9)
ax4.set_title('(d) Parental Education')
ax4.set_ylabel('Count')
for tick in ax4.get_xticklabels():
    tick.set_y(0.07)
ax4.legend()
# - Stratum
ax5 = fig.add_subplot(gs[2, :])
df['STRATUM'].value_counts().sort_index().plot(kind='bar', ax=ax5, color=blues(0.6), width=0.6)
ax5.set_title('(e) Stratum')
ax5.set_ylabel('Count')
ax5.set_xlabel('Socioeconomic Stratum')
ax5.set_xticklabels(df['STRATUM'].value_counts().sort_index().index, rotation=0, ha='center')
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=25)
plt.show()


## Dividing the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### MODEL BUILDING - SUPERVISED MACHINE LEARNING MODELS
## LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

# Pipeline
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# GridSearch
param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs', 'liblinear'],
    'classifier__max_iter': [100, 200, 500, 1000]
}
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
y_pred_lr = grid_search_lr.predict(X_test)
print("Best Parameters (Logistic Regression):", grid_search_lr.best_params_)

# Classification report
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))

# Confusion matrix
class_order = ['Low', 'Medium', 'High']
cm = confusion_matrix(y_test, y_pred_lr, labels=class_order)
print("Confusion Matrix (Logistic Regression):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
classes = ['Low', 'Medium', 'High']
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]
y_score = grid_search_lr.best_estimator_.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
from matplotlib.cm import get_cmap
blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature analysis
best_model_lr = grid_search_lr.best_estimator_.named_steps['classifier']
feature_names_lr = grid_search_lr.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
coefficients_lr = []
for class_idx, class_label in enumerate(best_model_lr.classes_):
    coeffs = best_model_lr.coef_[class_idx]
    df_tmp = pd.DataFrame({
        'Feature': feature_names_lr,
        'Coefficient': coeffs,
        'Class': class_label
    })
    coefficients_lr.append(df_tmp)
coefficients_lr = pd.concat(coefficients_lr)
ordered_classes = ['Low', 'Medium', 'High']
coefficients_lr['Class'] = pd.Categorical(coefficients_lr['Class'], categories=ordered_classes, ordered=True)
for cls in ordered_classes:
    print(f"\nTop 10 important features for class {cls}:")
    df_class = coefficients_lr[coefficients_lr['Class'] == cls]
    df_class = df_class.reindex(df_class['Coefficient'].abs().sort_values(ascending=False).index)
    print(df_class[['Feature', 'Coefficient']].head(10))

top_n = 10
fig, axes = plt.subplots(nrows=1, ncols=len(ordered_classes), figsize=(5 * len(ordered_classes), 6), sharey=True)
for idx, cls in enumerate(ordered_classes):
    df_class = coefficients_lr[coefficients_lr['Class'] == cls]
    df_class_sorted = df_class.reindex(df_class['Coefficient'].abs().sort_values(ascending=True).index)
    df_top = df_class_sorted.tail(top_n)
    colors = df_top['Coefficient'].apply(lambda x: '#67a9cf' if x > 0 else '#2166ac').tolist()
    sns.barplot(
        x='Coefficient',
        y='Feature',
        data=df_top,
        palette=colors,
        ax=axes[idx]
    )
    axes[idx].axvline(x=0, color='black', linewidth=0.8)
    axes[idx].set_title(f'Class: {cls}', fontsize=15, fontweight='bold')
    axes[idx].set_xlabel('Coefficient', fontsize=16)
    axes[idx].tick_params(axis='x', labelsize=14)
    axes[idx].tick_params(axis='y', labelsize=14)
    if idx == 0:
        axes[idx].set_ylabel('Feature')
    else:
        axes[idx].set_ylabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


## SUPPORT VECTOR MACHINE
from sklearn.svm import SVC

# Pipeline
pipeline_svm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))
])

# GridSearch
param_grid_svm = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
y_pred_svm = grid_search_svm.predict(X_test)
print("Best Parameters (SVM):", grid_search_svm.best_params_)

# Classification report
print("Classification Report (SVM):\n", classification_report(y_test, y_pred_svm))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_svm, labels=class_order)
print("Confusion Matrix (SVM):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# Feature analysis
best_model_svm = grid_search_svm.best_estimator_.named_steps['classifier']
if grid_search_svm.best_params_['classifier__kernel'] == 'linear':
    coeff_svm = best_model_svm.coef_[0]
    feature_names_svm = grid_search_svm.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
    coefficients_svm = pd.DataFrame({
        'Feature': feature_names_svm,
        'Coefficient': coeff_svm
    }).sort_values(by='Coefficient', ascending=False)
    print("Top 10 important features (SVM):\n", coefficients_svm.head(10))

    top_n = 10
    top_features = coefficients_svm.head(top_n).append(coefficients_svm.tail(top_n))
    plt.figure(figsize=(8, 6))
    colors = top_features['Coefficient'].apply(lambda x: '#67a9cf' if x > 0 else '#2166ac')
    sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.set_xlabel('Coefficient', fontsize=16)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("SVM with RBF kernel does not provide coefficients for feature importance.")


## K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier

# Pipeline
pipeline_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# GridSearch
param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 10],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}
grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
y_pred_knn = grid_search_knn.predict(X_test)
print("Best Parameters (KNN):", grid_search_knn.best_params_)

# Classification report
print("Classification Report (KNN):\n", classification_report(y_test, y_pred_knn))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_knn, labels=class_order)
print("Confusion Matrix (KNN):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
classes = ['Low', 'Medium', 'High']
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]
y_score = grid_search_knn.best_estimator_.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# We cannot use classical coefficients or feature importances, but we can estimate the relevance of the features using Permutation Feature Importance (standard solution for non-parametric models as this one).
from sklearn.inspection import permutation_importance

# Feature analysis
print("\nKNN - Feature Importance per class")
for cls in class_order:
    print(f"\nClass {cls}:")
    mask = (y_test == cls)
    result = permutation_importance(
        grid_search_knn, 
        X_test[mask], 
        y_test[mask],
        n_repeats=10, 
        random_state=42, 
        scoring='accuracy'
    )
    sorted_idx = result.importances_mean.argsort()[::-1]
    for i in sorted_idx:
        print(f"{X_test.columns[i]}: {result.importances_mean[i]:.4f}")

results_per_class = []
classes = np.unique(y_test)
top_n = 10
for cls in class_order:
    mask = (y_test == cls)
    result = permutation_importance(
        grid_search_knn,
        X_test[mask],
        y_test[mask],
        n_repeats=10,
        random_state=42,
        scoring='accuracy'
    )
    sorted_idx = result.importances_mean.argsort()[::-1][:top_n]
    df_top = pd.DataFrame({
        'Feature': X_test.columns[sorted_idx],
        'Importance': result.importances_mean[sorted_idx],
        'Class': cls
    })
    results_per_class.append(df_top)

df_pfi = pd.concat(results_per_class)
fig, axes = plt.subplots(nrows=1, ncols=len(class_order), figsize=(5 * len(class_order), 6), sharey=True)

for idx, cls in enumerate(class_order):
    df_class = df_pfi[df_pfi['Class'] == cls].sort_values('Importance', ascending=True)
    colors = ['#67a9cf' if val >= np.median(df_class['Importance']) else '#2166ac' for val in df_class['Importance']]
    sns.barplot(
        x='Importance',
        y='Feature',
        data=df_class,
        palette=colors,
        ax=axes[idx]
    )
    axes[idx].axvline(x=0, color='black', linewidth=0.8)
    axes[idx].set_title(f'Class: {cls}', fontsize=15, fontweight='bold')
    axes[idx].set_xlabel('Importance', fontsize=16)
    axes[idx].tick_params(axis='x', labelsize=14)
    axes[idx].tick_params(axis='y', labelsize=14)
    if idx == 0:
        axes[idx].set_ylabel('Feature')
    else:
        axes[idx].set_ylabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


## NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

# Pipeline
pipeline_nb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# GridSearch
param_grid_nb = {
    'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
}
grid_search_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=5, scoring='accuracy')
grid_search_nb.fit(X_train, y_train)
y_pred_nb = grid_search_nb.predict(X_test)
print("Best Parameters (Naive Bayes):", pipeline_nb.named_steps['classifier'].get_params())

# Classification report
print("Classification Report (Naive Bayes):\n", classification_report(y_test, y_pred_nb))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_nb, labels=class_order)
print("Confusion Matrix (Naive Bayes):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
classes = ['Low', 'Medium', 'High']
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]
y_score = grid_search_nb.best_estimator_.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature importance analysis is not applicable for Naive Bayes as it's a probabilistic model (and not a linear one).


## DECISION TREE
from sklearn.tree import DecisionTreeClassifier

# Pipeline
pipeline_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# GridSearch
param_grid_dt = {
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}
grid_search_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)
y_pred_dt = grid_search_dt.predict(X_test)
print("Best Parameters (Decision Tree):", grid_search_dt.best_params_)

# Classification report
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_dt, labels=class_order)
print("Confusion Matrix (Decision Tree):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_order,
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
classes = ['Low', 'Medium', 'High']
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]
y_score = grid_search_dt.best_estimator_.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature analysis
best_model_dt = grid_search_dt.best_estimator_.named_steps['classifier']
feature_names_dt = grid_search_dt.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
coefficients_dt = pd.DataFrame({
    'Feature': feature_names_dt,
    'Importance': best_model_dt.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)
coefficients_dt = coefficients_dt.sort_values(by='Importance', ascending=True)
print("Top 10 important features (Decision Tree):\n", coefficients_dt)
median_importance = coefficients_dt['Importance'].median()
colors = ['#67a9cf' if val >= median_importance else '#2166ac' for val in coefficients_dt['Importance']]
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=coefficients_dt,
    palette=colors
)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Feature', fontsize=16)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.show()


## RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

# Pipeline
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# GridSearch
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
y_pred_rf = grid_search_rf.predict(X_test)
print("Best Parameters (Random Forest):", grid_search_rf.best_params_)

# Classification report
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf, labels=class_order)
print("Confusion Matrix (Random Forest):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
y_test_bin = label_binarize(y_test, classes=class_order)
n_classes = y_test_bin.shape[1]
y_score = grid_search_rf.best_estimator_.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {class_order[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature analysis
best_model_rf = grid_search_rf.best_estimator_.named_steps['classifier']
feature_names_rf = grid_search_rf.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
importances_rf = pd.DataFrame({
    'Feature': feature_names_rf,
    'Importance': best_model_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Top 10 important features (Random Forest):\n", importances_rf.head(10))
top_rf = importances_rf.head(10).sort_values(by='Importance', ascending=True)
median_importance = top_rf['Importance'].median()
colors = ['#67a9cf' if val >= median_importance else '#2166ac'
          for val in top_rf['Importance']]
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=top_rf,
    palette=colors
)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Feature', fontsize=16)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.show()


### MODEL BUILDING - ENSEMBLE LEARNING APPROACHES
## GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingClassifier

# Pipeline
pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# GridSearch
param_grid_gb = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__learning_rate': [0.005, 0.01, 0.05, 0.1],
    'classifier__max_depth': [2, 3, 4, 5]
}
grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train, y_train)
y_pred_gb = grid_search_gb.predict(X_test)
print("Best Parameters (Gradient Boosting):", grid_search_gb.best_params_)

# Classification report
print("Classification Report (Gradient Boosting):\n", classification_report(y_test, y_pred_gb))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_gb, labels=class_order)
print("Confusion Matrix (Gradient Boosting):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
y_test_bin = label_binarize(y_test, classes=class_order)
n_classes = y_test_bin.shape[1]
y_score = grid_search_gb.best_estimator_.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {class_order[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature analysis
best_model_gb = grid_search_gb.best_estimator_.named_steps['classifier']
feature_names_gb = grid_search_gb.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
importances_gb = pd.DataFrame({
    'Feature': feature_names_gb,
    'Importance': best_model_gb.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Top 10 important features (Gradient Boosting):\n", importances_gb.head(10))
top_gb = importances_gb.head(10).sort_values(by='Importance', ascending=True)
median_importance = top_gb['Importance'].median()
colors = ['#67a9cf' if val >= median_importance else '#2166ac' for val in top_gb['Importance']]
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=top_gb,
    palette=colors
)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Feature', fontsize=16)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.show()


## ADABOOST
from sklearn.ensemble import AdaBoostClassifier

# Pipeline
pipeline_ab = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(algorithm='SAMME', random_state=42))
])

# GridSearch
param_grid_ab = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 1]
}
grid_search_ab = GridSearchCV(pipeline_ab, param_grid_ab, cv=5, scoring='accuracy')
grid_search_ab.fit(X_train, y_train)
y_pred_ab = grid_search_ab.predict(X_test)
print("Best Parameters (AdaBoost):", grid_search_ab.best_params_)

# Classification report
print("Classification Report (AdaBoost):\n", classification_report(y_test, y_pred_ab))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_ab, labels=class_order)
print("Confusion Matrix (AdaBoost):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
y_test_bin = label_binarize(y_test, classes=class_order)
n_classes = y_test_bin.shape[1]
y_score = grid_search_ab.best_estimator_.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {class_order[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature analysis
best_model_ab = grid_search_ab.best_estimator_.named_steps['classifier']
feature_names_ab = grid_search_ab.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
coefficients_ab = pd.DataFrame({
    'Feature': feature_names_ab,
    'Importance': best_model_ab.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Top 10 important features (AdaBoost):\n", coefficients_ab.head(10))
top_ab = coefficients_ab.head(10).sort_values(by='Importance', ascending=True)
median_val = top_ab['Importance'].median()
colors = ['#67a9cf' if val >= median_val else '#2166ac' for val in top_ab['Importance']]
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=top_ab,
    palette=colors
)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Feature', fontsize=16)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.show()


## CATBOOST
from catboost import CatBoostClassifier

# Pipeline
pipeline_cb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(random_state=42, verbose=0))
])

# GridSearch
param_grid_cb = {
    'classifier__iterations': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__depth': [5, 7, 10]
}
grid_search_cb = GridSearchCV(pipeline_cb, param_grid_cb, cv=5, scoring='accuracy')
grid_search_cb.fit(X_train, y_train)
y_pred_cb = grid_search_cb.predict(X_test)
print("Best Parameters (CatBoost):", grid_search_cb.best_params_)

# Classification report
print("Classification Report (CatBoost):\n", classification_report(y_test, y_pred_cb))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_cb, labels=class_order)
print("Confusion Matrix (CatBoost):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_order, 
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ROC Curve for One-vs-Rest classification
y_test_bin = label_binarize(y_test, classes=class_order)
n_classes = y_test_bin.shape[1]
best_model_cb = grid_search_cb.best_estimator_
y_score_cb = best_model_cb.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_cb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
blues = get_cmap('Blues')
colors = [blues(0.6), blues(0.75), blues(0.9)]
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {class_order[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.legend(loc="upper left", fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature analysis
best_model_cb = grid_search_cb.best_estimator_.named_steps['classifier']
feature_names_cb = grid_search_cb.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
importances_norm = best_model_cb.feature_importances_ / best_model_cb.feature_importances_.sum()
coefficients_cb = pd.DataFrame({
    'Feature': feature_names_cb,
    'Normalized Importance': importances_norm
}).sort_values(by='Normalized Importance', ascending=False)
print("Top 10 important features (CatBoost - Normalized):\n", coefficients_cb.head(10))
top_cb = coefficients_cb.head(10).sort_values(by='Normalized Importance', ascending=True)
median_val = top_cb['Normalized Importance'].median()
colors = ['#67a9cf' if val >= median_val else '#2166ac' for val in top_cb['Normalized Importance']]
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Normalized Importance',
    y='Feature',
    data=top_cb,
    palette=colors
)
plt.xlabel('Normalized Importance', fontsize=16)
plt.ylabel('Feature', fontsize=16)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.show()


## STACKING
from sklearn.ensemble import StackingClassifier

# Defining the base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
]

# Defining the meta-learner
meta_learner = LogisticRegression()

# Creating the stacking classifier (with 5-fold cross-validation)
stacked_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

# Pipeline
pipeline_stack = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacked_model)
])
pipeline_stack.fit(X_train, y_train)
y_pred_stack = pipeline_stack.predict(X_test)

# Classification report
print("Classification Report (Stacking):\n", classification_report(y_test, y_pred_stack))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_stack, labels=class_order)
print("Confusion Matrix (Stacking):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_order,
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()


## VOTING - MAJORITY VOTE
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('gb', best_model_gb),
        ('svm', best_model_svm),
        ('ab', best_model_ab)
    ],
    voting='hard'
)

# Pipeline
pipeline_voting = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])
pipeline_voting.fit(X_train, y_train)
y_pred_voting = pipeline_voting.predict(X_test)

# Classification report
print("Classification Report (Voting):\n", classification_report(y_test, y_pred_voting))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_voting, labels=class_order)
print("Confusion Matrix (Voting):\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_order,
            yticklabels=class_order,
            cbar=False, linewidths=0.5, linecolor='blue',
            annot_kws={"size": 18})
plt.xlabel("Predicted Label", fontsize = 20, labelpad = 15)
plt.ylabel("True Label", fontsize = 20, labelpad = 15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()