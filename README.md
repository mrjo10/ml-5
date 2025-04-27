# 1. Imports and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
%matplotlib inline

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

df.info()
df.describe()

sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
plt.suptitle('Pairplot of Iris Features', y=1.02)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
features = iris.feature_names
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(x='species', y=feat, data=df, ax=ax)
    ax.set_title(f'Boxplot of {feat}')
fig.tight_layout()
plt.show()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
# PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = df['species']
sns.scatterplot(x='PC1', y='PC2', hue='species', data=df_pca, s=60)
plt.title('PCA Projection of Iris Data')
plt.show()

# Features and target
X = df[iris.feature_names]
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(shap_values[2].shape)
print(X_test.shape)

import shap
import matplotlib.pyplot as plt

# Use a general explainer
explainer = shap.Explainer(rf, X_train)  

# Get SHAP values for test set
shap_values = explainer(X_test)

# Check shape
print(shap_values.values.shape)  
shap_values_class2 = shap_values.values[:, :, 2]  

print(shap_values_class2.shape)  

# 2. Now plot
shap.summary_plot(shap_values_class2, X_test, feature_names=iris.feature_names)

plt.show()

# LIME explanation for first test instance
explainer_lime = LimeTabularExplainer(
    X_train_scaled, feature_names=iris.feature_names,
    class_names=iris.target_names, mode='classification'
)
exp = explainer_lime.explain_instance(
    X_test_scaled[0], rf.predict_proba, num_features=4
)
exp.show_in_notebook(show_table=True)

# Permutation importance
perm_imp = permutation_importance(rf, X_test_scaled, y_test, n_repeats=15, random_state=42)
imp_means = perm_imp.importances_mean
indices = np.argsort(imp_means)
plt.barh(np.array(iris.feature_names)[indices], imp_means[indices])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance')
plt.show()


