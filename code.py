"""
Load Data
"""
import pandas as pd
df=pd.read_csv("your path")

"""
Data Cleansing - Feature Engineering
"""
#Cleaning path column and delete unwanted devices
df['path'] = df['path'].apply(lambda x: x[1:-1])
df['path'] = df['path'].str.replace(r'\[|\]', '', regex=True)
print(df.shape)
df = df[~df['path'].str.contains('Unknown')]
df = df[~df['path'].str.contains('Robot')]
print(df.shape)

#Creating Sequence
device_to_index = {}
def convert_to_numerical(path):
    numerical_sequence = []
    steps = path.split(', ')
    for step in steps:
        if not step.isdigit():
            if step not in device_to_index:
                device_to_index[step] = len(device_to_index) + 1
            numerical_sequence.append(device_to_index[step])
    return numerical_sequence
  
# Apply the function to each row in the 'path' column
df['numerical_sequence'] = df['path'].apply(convert_to_numerical)
# Calculate lenght of each path
df['path_length'] = df['numerical_sequence'].apply(len)
print(device_to_index)
print(df.shape)
df.head()

#count of devices in each row
# Remove numeric values from the devices set
all_devices = [device for device in device_to_index if not device.isdigit()]

# Create columns for each device and initialize them with zeros
for device in all_devices:
    df[device] = 0
  
# Define a function to count devices in each row and update the corresponding column
def count_devices(row):
    devices = row['path'].split(', ')
    for device in devices:
        if not device.isdigit():
            row[device] += 1
    return row

# Apply the function to each row
df = df.apply(count_devices, axis=1)
print(df.shape)
df.head()

#Add necessary features
df['roi'] = df['sales'] / df['cost']
print(df.isna().sum())
df['roi'].fillna(0, inplace=True)
print("------------------------------------------------")
print(df.isna().sum())
print("------------------------------------------------")
df['roi per impression'] =df['roi']/df['impressions']
print(df.shape)

# Define  ROI categories based on buyer or non buyer
bins = [-1, 0,   float('inf')]
labels = ['non-buyer', 'buyer']
df = df.sort_values(by='roi')
df['roi_category'] = pd.cut(df['roi'], bins=bins, labels=labels, include_lowest=True)
print(df['roi_category'].value_counts())
df.head()

# Define  ROI categories based on low and high roi
bins = [-1, 0,15,  float('inf')]
labels = ['non-buyer', 'low_roi',  'high_roi']
df = df.sort_values(by='roi')
df['roi_level'] = pd.cut(df['roi'], bins=bins, labels=labels, include_lowest=True)
print(df['roi_level'].value_counts())
df.head()

"""
Data Visualization
"""
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import numpy as np

#Correlation and heatmap
# Calculate correlation matrix
correlation_matrix = df[['impressions', 'roi', 'cost', 'sales', 'roi per impression']].corr()
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
# Show the plot
plt.show()

#Scatter Plot of Cost vs. Sales
plt.scatter(df['cost'], df['sales'])
plt.title('Scatter Plot of Cost vs. Sales')
plt.xlabel('Cost')
plt.ylabel('Sales')
plt.show()

#Financial Summary with a bar plot
df['profit'] = df['sales'] - df['cost']
total_spent = df['cost'].sum()
total_earned = df['sales'].sum()
total_profit = df['profit'].sum()

plt.figure(figsize=(8, 5))
categories = ['Money Spent', 'Money Earned', 'Profit']
amounts = [total_spent, total_earned, total_profit]
colors = ['red', 'green', 'blue']
plt.bar(categories, amounts, color=colors)
plt.ylabel('Amount ($)')
plt.title('Financial Summary')
# Add the amounts on top of the bars
for i, v in enumerate(amounts):
    plt.text(i, v + 1000, f"${v:,.2f}", color='black', ha='center', va='bottom', fontweight='bold')
plt.show()

#Top 5 ROI Paths
# Sort the DataFrame by ROI in descending order and select the top 5 rows
top_paths = df.nlargest(5, 'roi')
# Create a figure
fig, ax = plt.subplots()
# Set the y-axis position for each bar
y_pos = range(len(top_paths))
# Create horizontal bar chart
ax.barh(y_pos, top_paths['roi'], align='center', color='skyblue')
# Wrap y-axis labels into two lines
ax.set_yticks(y_pos)
ax.set_yticklabels([textwrap.fill(path, 85) for path in top_paths['path']])
ax.invert_yaxis()  # Invert y-axis for better readability
# Add labels and title
plt.xlabel('ROI')
plt.ylabel('Path')
plt.title('Top 5 ROI Paths')
# Add the exact 'roi per impression' values to each column
for i, roi_value in enumerate(top_paths['roi']):
    plt.text(roi_value, i, f'{roi_value:.2f}', va='center', ha='right', color='black', fontsize=8)
# Show the plot
plt.show()

#Top 5 ROI per impression Paths
# Sort the DataFrame by ROI per impression in descending order and select the top 5 rows
top_paths = df.nlargest(5, 'roi per impression')
# Create a figure
fig, ax = plt.subplots()
# Set the y-axis position for each bar
y_pos = range(len(top_paths))
# Create horizontal bar chart
ax.barh(y_pos, top_paths['roi per impression'], align='center', color='skyblue')
# Wrap y-axis labels into two lines
ax.set_yticks(y_pos)
ax.set_yticklabels([textwrap.fill(path, 85) for path in top_paths['path']])
ax.invert_yaxis()  # Invert y-axis for better readability
# Add labels and title
plt.xlabel('ROI per impression')
plt.title('Top 5 ROI per impression Paths')
# Add the exact 'roi per impression' values to each column
for i, roi_value in enumerate(top_paths['roi per impression']):
    plt.text(roi_value, i, f'{roi_value:.2f}', va='center', ha='right', color='black', fontsize=8)
# Show the plot
plt.show()

#Top 5 impression
# Sort the DataFrame by impressions in descending order and select the top 5 rows
top_paths = df.nlargest(5, 'impressions')
# Create a figure
fig, ax = plt.subplots()
# Set the y-axis position for each bar
y_pos = range(len(top_paths))
# Create horizontal bar chart
ax.barh(y_pos, top_paths['impressions'], align='center', color='skyblue')
# Wrap y-axis labels into two lines
ax.set_yticks(y_pos)
ax.set_yticklabels([textwrap.fill(path, 85) for path in top_paths['path']])
ax.invert_yaxis()  # Invert y-axis for better readability
# Add labels and title
plt.xlabel('Impressions')
plt.title('Top 5 Impression Paths')
# Add the exact 'impressions' values to each column
for i, impressions_value in enumerate(top_paths['impressions']):
    formatted_value = '{:,}'.format(impressions_value)  # Format with commas
    plt.text(impressions_value, i, f'{formatted_value}', va='center', ha='right', color='black', fontsize=8)
# Show the plot
plt.show()

#Top 5 impression vs ROI
# Sort the DataFrame by Impressions in descending order and select the top 5 rows
top_paths = df.nlargest(5, 'impressions')
# Create a figure
fig, ax1 = plt.subplots(figsize=(10, 6))
# Set the x-axis position for each bar
x_pos = np.arange(len(top_paths))
# Create clustered column chart for Impressions
bars = ax1.bar(x_pos, top_paths['impressions'], width=0.4, align='center', color='skyblue', label='Impressions')
# Add labels and title
ax1.set_ylabel('Impressions')
ax1.set_title('Top 5 Paths by Impressions')
# Set x-axis ticks and labels
ax1.set_xticks(x_pos)
ax1.set_xticklabels([textwrap.fill(path, 20) for path in top_paths['path']], rotation=45, ha='right')
# Create a secondary y-axis for ROI
ax2 = ax1.twinx()
ax2.plot(x_pos, top_paths['roi'], color='orange', marker='o', label='ROI')
# Set y-axis label for ROI
ax2.set_ylabel('ROI')
# Combine the legends and place them at the top-middle of the plot
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')
# Add ROI and Impressions numbers for each column
for i, bar in enumerate(bars):
    yval = bar.get_height()
    formatted_impressions = '{:,}'.format(int(yval))  # Add thousand separators
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1000, formatted_impressions, va='bottom', ha='center', color='black', fontsize=8)
    ax2.text(bar.get_x() + bar.get_width()/2, top_paths['roi'].iloc[i], f"{top_paths['roi'].iloc[i]:.2f}", va='bottom', ha='center', color='red', fontsize=8)
# Show the plot
plt.tight_layout()
plt.show()

"""
Data Preparation
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

#Equalization buyer and non-buyer path
# Separate paths with ROI equal to zero and non-zero ROI
zero_roi_paths = df[df['roi'] == 0]
non_zero_roi_paths = df[df['roi'] != 0]
# Sample the same number of paths with non-zero ROI as there are paths with zero ROI
num_samples = len(non_zero_roi_paths)
sampled_non_zero_roi_paths = zero_roi_paths.sample(n=num_samples,replace=True)#, random_state=42)
# Combine the sampled paths
balanced_df = pd.concat([non_zero_roi_paths, sampled_non_zero_roi_paths])
# Now, 'balanced_df' contains an equal number of paths with ROI equal to zero and non-zero ROI
print(df['roi_category'].value_counts())
print("----------")
print(balanced_df['roi_category'].value_counts())

#Separating features and Target for buyer and non-buyer
X = balanced_df['numerical_sequence']
y = balanced_df['roi_category']

#Encoding Target and Padding sequences for buyer and non-buyer
# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Pad sequences to have a fixed length
max_sequence_length = 50 #max(len(seq) for seq in X)
X_padded = pad_sequences(X, maxlen=max_sequence_length, padding='post')

#Equalization roi level
non_zero_roi_paths2 = df[df['roi_level'] != 'non-buyer']
non_zero_roi_paths2['roi_level'] = non_zero_roi_paths2['roi_level'].str.strip()
zero_roi_paths2 = df[df['roi'] == 0]
balanced_df2 = pd.concat([non_zero_roi_paths2])#, sampled_zero_roi_paths2])
print(df['roi_level'].value_counts())
print("----------")
print(balanced_df2['roi_level'].value_counts())

#Separating features and Target for ROI levels
X2 = balanced_df2['numerical_sequence']
y2 = balanced_df2['roi_level']

#Encoding Target and Padding sequences for ROI levels
# Encode the target labels
label_encoder2 = LabelEncoder()
y_encoded2 = label_encoder2.fit_transform(y2)
# Pad sequences to have a fixed length
max_sequence_length = 50 #max(len(seq) for seq in X)
X_padded2 = pad_sequences(X2, maxlen=max_sequence_length, padding='post')

"""
Training Model (buyer /non-buyer)
"""
pip install scikit-optimize
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Define LSTM model
def create_lstm_model(units=64, hidden_units=32, dropout_rate=0.2, optimizer='adam'):
    model = Sequential([
        Embedding(input_dim=len(device_to_index)+1, output_dim=50, input_length=max_sequence_length),
        LSTM(units, dropout=dropout_rate),
        Dense(hidden_units, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model on your data
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_encoded, test_size=0.2)#, random_state=42)

#Hyperparameter optimization and selection
# Define hyperparameters search space
space = [
    Integer(32, 128, name='units'),
    Integer(16, 64, name='hidden_units'),
    Real(0.1, 0.5, name='dropout_rate'),
    Categorical(['adam', 'rmsprop'], name='optimizer')
]
@use_named_args(space)
def objective(units, hidden_units, dropout_rate, optimizer):
    model = create_lstm_model(units=units, hidden_units=hidden_units, dropout_rate=dropout_rate, optimizer=optimizer)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return -accuracy  # Minimize, so negate the accuracy
# Perform Bayesian optimization
result = gp_minimize(objective, space, n_calls=20, random_state=42)
# Get the best hyperparameters
best_hyperparameters = dict(zip(['units', 'hidden_units', 'dropout_rate', 'optimizer'], result.x))
print("Best Hyperparameters:", best_hyperparameters)
# Define your LSTM model with the best hyperparameters
best_model = create_lstm_model(**best_hyperparameters)
best_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)
# Evaluate the best model
loss, accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print(f'Accuracy of the best model on the test data: {accuracy*100:.2f}%')

#Evaluate The Performance
y_pred = best_model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=-1)
# If y_pred_labels are encoded, you can decode them back to their original labels
y_pred_labels = label_encoder.inverse_transform(y_pred_labels)
y_original = label_encoder.inverse_transform(y_val)
results = pd.DataFrame({'Predicted': y_pred_labels, 'Actual': y_original})
results.head()
# Calculate confusion matrix
confusion_mtx = confusion_matrix(y_original, y_pred_labels, labels=['buyer', 'non-buyer'])
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_mtx)
# Print classification report
print("\nClassification Report:")
print(classification_report(y_original, y_pred_labels))

"""
Training Model (ROI level)
"""
# Define LSTM model
def create_lstm_model(units=64, hidden_units=32, dropout_rate=0.2, optimizer='adam'):
    model = Sequential([
        Embedding(input_dim=len(device_to_index)+1, output_dim=50, input_length=max_sequence_length),
        LSTM(units, dropout=dropout_rate),
        Dense(hidden_units, activation='relu'),
        Dense(len(label_encoder2.classes_), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
  X_train2, X_val2, y_train2, y_val2 = train_test_split(X_padded2, y_encoded2, test_size=0.2)#, random_state=42)

#Hyperparameter optimization and selection
# Define hyperparameters search space
space = [
    Integer(32, 128, name='units'),
    Integer(16, 64, name='hidden_units'),
    Real(0.1, 0.5, name='dropout_rate'),
    Categorical(['adam', 'rmsprop'], name='optimizer')
]
@use_named_args(space)
def objective(units, hidden_units, dropout_rate, optimizer):
    model = create_lstm_model(units=units, hidden_units=hidden_units, dropout_rate=dropout_rate, optimizer=optimizer)
    model.fit(X_train2, y_train2, epochs=5, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_val2, y_val2, verbose=0)
    return -accuracy  # Minimize, so negate the accuracy
# Perform Bayesian optimization
result2 = gp_minimize(objective, space, n_calls=20, random_state=42)
# Get the best hyperparameters
best_hyperparameters2 = dict(zip(['units', 'hidden_units', 'dropout_rate', 'optimizer'], result2.x))
print("Best Hyperparameters:", best_hyperparameters2)
# Define your LSTM model with the best hyperparameters
best_model2 = create_lstm_model(**best_hyperparameters2)
# Evaluate the best model
loss, accuracy = best_model2.evaluate(X_val2, y_val2, verbose=0)
print(f'Accuracy of the best model on the test data: {accuracy*100:.2f}%')

#Evaluate The Performance
y_pred2 = best_model2.predict(X_val2)
y_pred_labels2 = np.argmax(y_pred2, axis=1)
# If y_pred_labels are encoded, you can decode them back to their original labels
y_pred_labels2 = label_encoder2.inverse_transform(y_pred_labels2)
y_original2 = label_encoder2.inverse_transform(y_val2)
results2 = pd.DataFrame({'Predicted': y_pred_labels2, 'Actual': y_original2})
results2.head()
# Calculate confusion matrix
confusion_mtx2 = confusion_matrix(y_original2, y_pred_labels2)
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_mtx2)
# Print classification report
print("\nClassification Report:")
print(classification_report(y_original2, y_pred_labels2))

"""
Machine learning models
"""
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
balanced_df3=balanced_df.copy()
scaler = StandardScaler()
balanced_df3['roi'] = scaler.fit_transform(balanced_df3[['roi']])
balanced_df3['impressions'] = scaler.fit_transform(balanced_df3[['impressions']])
balanced_df3['path_length'] = scaler.fit_transform(balanced_df3[['path_length']])
y3=balanced_df3['roi']
X3=balanced_df3[['impressions','path_length','Phone','PC','TV','Tablet']]
# Split the data into training and testing sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]}
# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# Fit the Grid Search to find the best model
grid_search.fit(X_train3, y_train3)
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
# Get the best model
best_rf_model = grid_search.best_estimator_
# Make predictions
y_pred3 = best_rf_model.predict(X_test3)
# Calculate evaluation metrics
mse = mean_squared_error(y_test3, y_pred3)
r2 = r2_score(y_test3, y_pred3)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test3, y_pred3)
# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Get feature importances
feature_importances = best_rf_model.feature_importances_
# Print feature importances alongside corresponding feature names
for feature, importance in zip(X3.columns, feature_importances):
    print(f"{feature}: {importance}")

#Ridge and Lasso
# Initialize Ridge Regression
ridge_model = Ridge()
# Define the parameter grid for Grid Search
ridge_param_grid = {'alpha': [0.1, 1, 10, 100]}
# Initialize Grid Search with cross-validation
ridge_grid_search = GridSearchCV(ridge_model, ridge_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# Fit the Grid Search to find the best model
ridge_grid_search.fit(X_train3, y_train3)
# Get the best Ridge model
best_ridge_model = ridge_grid_search.best_estimator_
# Initialize Lasso Regression
lasso_model = Lasso()
# Define the parameter grid for Grid Search
lasso_param_grid = {'alpha': [0.1, 1, 10, 100]}
# Initialize Grid Search with cross-validation
lasso_grid_search = GridSearchCV(lasso_model, lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# Fit the Grid Search to find the best model
lasso_grid_search.fit(X_train3, y_train3)
# Get the best Lasso model
best_lasso_model = lasso_grid_search.best_estimator_
# Initialize Linear Regression
lr_model = LinearRegression()
# Fit the model
lr_model.fit(X_train3, y_train3)
# Predict using all models
y_pred_ridge = best_ridge_model.predict(X_test3)
y_pred_lasso = best_lasso_model.predict(X_test3)
y_pred_lr = lr_model.predict(X_test3)
# Calculate evaluation metrics for Ridge
mse_ridge = mean_squared_error(y_test3, y_pred_ridge)
r2_ridge = r2_score(y_test3, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_test3, y_pred_ridge)
ridge_coefficients = best_ridge_model.coef_
# Calculate evaluation metrics for Lasso
mse_lasso = mean_squared_error(y_test3, y_pred_lasso)
r2_lasso = r2_score(y_test3, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test3, y_pred_lasso)
lasso_coefficients = best_lasso_model.coef_
# Calculate evaluation metrics for Linear Regression
mse_lr = mean_squared_error(y_test3, y_pred_lr)
r2_lr = r2_score(y_test3, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test3, y_pred_lr)
lr_coefficients = lr_model.coef_

# Print evaluation metrics
print("Ridge Regression:")
print(f"Mean Squared Error (MSE): {mse_ridge}")
print(f"R-squared (R2): {r2_ridge}")
print(f"Root Mean Squared Error (RMSE): {rmse_ridge}")
print(f"Mean Absolute Error (MAE): {mae_ridge}")
print("\n")
print("Ridge Coefficients:")
for feature, coef in zip(X3.columns, ridge_coefficients):
    print(f"{feature}: {coef}")
print("\n")
print("Lasso Regression:")
print(f"Mean Squared Error (MSE): {mse_lasso}")
print(f"R-squared (R2): {r2_lasso}")
print(f"Root Mean Squared Error (RMSE): {rmse_lasso}")
print(f"Mean Absolute Error (MAE): {mae_lasso}")
print("\n")
print("Lasso Coefficients:")
for feature, coef in zip(X3.columns, lasso_coefficients):
    print(f"{feature}: {coef}")
print("\n")
print("Linear Regression:")
print(f"Mean Squared Error (MSE): {mse_lr}")
print(f"R-squared (R2): {r2_lr}")
print(f"Root Mean Squared Error (RMSE): {rmse_lr}")
print(f"Mean Absolute Error (MAE): {mae_lr}")
print("\n")
print("Linear Regression Coefficients:")
for feature, coef in zip(X3.columns, lr_coefficients):
    print(f"{feature}: {coef}")
