import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Import joblib for model serialization



FEATURE_NAMES_FILENAME = "kddcup.names"
TRAINING_DATA_FILENAME = "kddcup.csv"
TRAINING_ATTK_FILENAME = "attacktypes"

with open(FEATURE_NAMES_FILENAME, 'r') as kddcup_names:
    label_names = kddcup_names.readline().strip().split(",")
    
    _remainder = kddcup_names.readlines()

    feature_names = [l.split(": ")[0] for l in _remainder] + ["target"]
    feature_types = [l.split(": ")[1].split(".")[0] for l in _remainder] + ["continuous"]

with open(TRAINING_ATTK_FILENAME, 'r') as training_attack_names_f:
    training_attack_names = training_attack_names_f.read().strip().split("\n")

training_attack_names


# In[3]:


df = pd.read_csv(TRAINING_DATA_FILENAME, names=feature_names)
df.drop_duplicates(subset=None, keep='first', inplace=True)

print("shape",df.shape)


# In[4]:


# These are "categorical" variables to convert to one-hot:
symbolic_features = [
    f for (t, f) in zip(feature_types, feature_names) 
    if (t == 'symbolic' and not (
        df.dtypes[f] == int and df.nunique()[f] <= 2
    ) and t != 'count')
]
print("shape2",symbolic_features)


# In[5]:


continuous_features = [
    f for (t, f) in zip(feature_types, feature_names) 
    if t == "continuous"
]
print("shape3",continuous_features)


# In[6]:


print("shape4",df.head())


# In[7]:


print("shape5",df.target.value_counts())


# In[8]:


print("shape6",df.info())


# In[9]:


print("shape7",df.nunique())







df.drop('num_outbound_cmds', axis=1, inplace=True)
feature_names.remove('num_outbound_cmds')


# In[11]:


# get splitted DFs per label:

intact_df_per_label = dict([i for i in df.copy().groupby(df['target'])])
with_normal_df_per_label = intact_df_per_label.copy()

df_normal = with_normal_df_per_label["normal."].copy()
df_without_normal = df[df.target != "normal."]
del with_normal_df_per_label["normal."]

# Adding the normal data next to every df:
for label, label_df in with_normal_df_per_label.items():
    label_df = pd.concat([label_df, df_normal])
    with_normal_df_per_label[label] = label_df

with_normal_df_per_label.keys()


# ## Plotting feature distributions

# In[12]:


#import plotly.graph_objects as go
#import plotly.express as px

print("hi")
# ### Categorical variables

# In[13]:


df_categorical_cols = df.select_dtypes(include='object').columns

print("hi5",df_categorical_cols)
# In[14]:


# for feature in df_categorical_cols:
#     fig = px.histogram(df, x=feature, log_y=True, color="target", color_discrete_sequence=px.colors.qualitative.Light24)
#     fig.update_layout(xaxis={'categoryorder':'total ascending'})
#     fig.show(renderer="svg", width=900, height=600)

# Iterate over categorical columns
for feature in df_categorical_cols:
    # Count the occurrences of each category
    category_counts = df[feature].value_counts()
    if feature == "service":
        continue  # Skip the "service" column
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))  # Increase figure width to accommodate longer labels
    bars = plt.bar(category_counts.index, category_counts.values)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(f'Histogram of {feature}')
    plt.xticks(rotation=45, fontsize=8)  # Adjust rotation angle and font size
    
    # Add text annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()
print("hi2")
# ### Continuous variables

# In[15]:


df_continuous_cols = df.select_dtypes(exclude='object').columns


# In[16]:


# for feature_name in df_continuous_cols:
#     fig = px.histogram(df, x=feature_name, log_y=True, color="target", color_discrete_sequence=px.colors.qualitative.Light24)
#     fig.show(renderer="svg", width=900, height=600)



# Iterate over continuous columns
# for feature_name in df_continuous_cols:
#     # Create a histogram
#     plt.figure(figsize=(10, 6))
#     plt.hist(df[feature_name], bins=20, color='skyblue', edgecolor='black')
#     plt.xlabel(feature_name)
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of {feature_name}')
#     plt.grid(True)
#     plt.show()

# ### Plotting feature correlation matrices

# In[17]:


def log1p_sorted_abs_feature_correlation_matrix(df):
    corr_df = df.corr()
    corr_df.dropna(axis=0, how='all', inplace=True)
    corr_df.dropna(axis=1, how='all', inplace=True)
    corr_df = corr_df.abs()

    # sort both axis:
    corr_df = corr_df.pipe(lambda df: df.loc[:, df.sum().sort_values(ascending=False).index]).transpose(
        ).pipe(lambda df: df.loc[:, df.sum().sort_values(ascending=False).index]).transpose()

    corr_df = corr_df.apply(lambda x: np.log1p(x))

    return corr_df

# def plot_modified_correlation_matrix(df, title):
#     fig = px.imshow(log1p_sorted_abs_feature_correlation_matrix(df), title=title)
#     fig.show(renderer='svg', width=900, height=900)
import seaborn as sns

def plot_modified_correlation_matrix(df, title):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Compute the correlation matrix
    corr_matrix = numeric_df.corr()

    # Plot the correlation matrix using seaborn heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()

# Plot the correlation matrix for the entire DataFrame
plot_modified_correlation_matrix(df, "Feature Correlation Matrix")

















from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def plot_feature_importance(df, title, title2=None, max_tree_depth=15, plot_top_2=False):
    clf = DecisionTreeClassifier(max_depth=max_tree_depth)
    X = df[df_continuous_cols].values[:, :-1]
    y = df.values[:, -1]
    clf = clf.fit(X, y)

    feature_importance_df = pd.DataFrame(list(zip(clf.feature_importances_, feature_names)), columns=["feature_importance", "feature_name"])
    feature_importance_df = feature_importance_df.sort_values(by='feature_importance', ascending=False)
    useless_features = list(feature_importance_df[feature_importance_df['feature_importance'] == 0]['feature_name'])
    feature_importance_df = feature_importance_df[feature_importance_df['feature_importance'] != 0]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(feature_importance_df['feature_name'], feature_importance_df['feature_importance'], log=True)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    
    # Add numbers to the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}', ha='center', va='bottom')
    
    plt.show()
    
    print("The following features were dropped:")
    print(useless_features)
    
    top_features = feature_importance_df['feature_name']  # TODO: if feature in continuous_features
    if plot_top_2:
        if len(top_features) >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(df[top_features.iloc[0]], df[top_features.iloc[1]], c=df['target'])
            plt.title(title2)
            plt.xlabel(top_features.iloc[0])
            plt.ylabel(top_features.iloc[1])
            plt.colorbar(label='target')
            
            # Add annotations to the scatter plot
            for i in range(len(df)):
                plt.text(df[top_features.iloc[0]].iloc[i], df[top_features.iloc[1]].iloc[i], df['target'].iloc[i])
            
            plt.show()
        else:
            print("Skipping feature importance scatterplot as there are less than 2 features.")
plot_feature_importance(df, "Overall feature importance from simple decision tree.", "Overall top two features' scatterplot.", plot_top_2=True)



