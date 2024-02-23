# Import necessary libraries
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Define column names for the dataset
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
             "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
             "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
             "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
             "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
             "label"]

# Read the training dataset
kdd_data_10percent = pd.read_csv("kddcup.csv", names=col_names)

# Display descriptive statistics of the dataset
print(kdd_data_10percent.describe())

# Display counts of each unique value in the 'label' column
print(kdd_data_10percent['label'].value_counts())

# Select a subset of numerical features
num_features = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                 "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
                 "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
                 "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                 "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

# Convert selected features to float
features = kdd_data_10percent[num_features].astype(float)

# Display descriptive statistics of the selected features
print(features.describe())

# Rescale the selected features using Min-Max scaling
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Display descriptive statistics of the scaled features
print(pd.DataFrame(features_scaled, columns=num_features).describe())

# Label encoding: Replace 'normal.' with 'attack.'
labels = kdd_data_10percent['label'].copy()
labels[labels != 'normal.'] = 'attack.'

# Display counts of each unique value in the modified 'label' column
print(labels.value_counts())

# Read the test dataset
kdd_data_corrected = pd.read_csv("corrected_for_test", header=None, names=col_names)

# Display counts of each unique value in the 'label' column of the test dataset
print(kdd_data_corrected['label'].value_counts())

# Label encoding for the test dataset: Replace 'normal.' with 'attack.'
kdd_data_corrected['label'][kdd_data_corrected['label'] != 'normal.'] = 'attack.'

# Display counts of each unique value in the modified 'label' column of the test dataset
print(kdd_data_corrected['label'].value_counts())

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Convert numerical features of the test dataset to float
kdd_data_corrected[num_features] = kdd_data_corrected[num_features].astype(float)

# Rescale the numerical features of the test dataset using the same scaler
kdd_data_corrected[num_features] = scaler.fit_transform(kdd_data_corrected[num_features])

# Train-Test Split
features_train, features_test, labels_train, labels_test = train_test_split(
    kdd_data_corrected[num_features], kdd_data_corrected['label'], test_size=0.1)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=0)
t0 = time()
clf.fit(features_train, labels_train)
tt = time() - t0
print("Classifier trained in {} seconds.".format(round(tt, 3)))

# Make predictions on the test data
pred = clf.predict(features_test)
tt = time() - t0
print("Predicted in {} seconds".format(round(tt, 3)))

# Calculate and display the accuracy
acc = accuracy_score(pred, labels_test)
print("Accuracy is {}.".format(round(acc, 4)))

# Confusion Matrix
cm = confusion_matrix(labels_test, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['normal', 'attack'], yticklabels=['normal', 'attack'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


plt.figure(figsize=(10, 6))
bar_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
bar_values = [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]
bars = sns.barplot(x=bar_labels, y=bar_values, palette="Blues")
plt.title('Confusion Matrix Counts')

# Add numbers to the bars
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    plt.annotate(f'{height}', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.show()

# Classification Report
acc = accuracy_score(pred, labels_test)
accuracy_message = "Accuracy is {}.".format(round(acc, 4))
print(accuracy_message)

# Classification Report
classification_rep = classification_report(labels_test, pred)
classification_message = "Classification Report:\n" + classification_rep
print(classification_message)

# Save accuracy and classification report to a file
with open('classification_results.txt', 'w') as file:
    file.write(accuracy_message + "\n\n")
    file.write(classification_message)


# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import joblib  # Import joblib for model serialization
# model_filename = "random_forest_kdd99_model.h5"
# joblib.dump(clf, model_filename)
# print(f"Model saved as {model_filename}")


# cm = confusion_matrix(labels_test, pred)
# plt.matshow(cm, cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.show()


# cm = confusion_matrix(labels_test, pred)
# sns.clustermap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['normal', 'attack'], yticklabels=['normal', 'attack'])
# plt.title('Confusion Matrix')
# plt.show()





# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# Generate classification report
# report_dict = classification_report(labels_test, pred, output_dict=True)

# Convert the dictionary to a DataFrame
# report_df = pd.DataFrame(report_dict).transpose()

# Plot precision, recall, and f1-score
# plt.figure(figsize=(10, 6))
# report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', colormap='viridis')
# plt.title('Classification Report Metrics')
# plt.xlabel('Class')
# plt.ylabel('Score')
# plt.legend(loc='upper right')
# plt.show()