import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

class AccountFeatures:
    def __init__(self, account_no,
                 avg_transactions_per_month, avg_withdrawals_per_month, avg_deposits_per_month,
                 avg_withdrawal_amount, avg_deposit_amount,
                 avg_balance_amount_upper, avg_balance_amount_lower, anomaly_score):
        self.account_no = account_no
        self.avg_transactions_per_month = avg_transactions_per_month
        self.avg_withdrawals_per_month = avg_withdrawals_per_month
        self.avg_deposits_per_month = avg_deposits_per_month
        self.avg_withdrawal_amount = avg_withdrawal_amount
        self.avg_deposit_amount = avg_deposit_amount
        self.avg_balance_amount_upper = avg_balance_amount_upper
        self.avg_balance_amount_lower = avg_balance_amount_lower
        self.anomaly_score = anomaly_score

# Replace 'your_excel_file.xlsx' with the actual path to your Excel file
excel_file_path = 'iit_data.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Handling NaN values in 'DEPOSIT AMT' and 'WITHDRAWAL AMT'
df['DEPOSIT AMT'] = df['DEPOSIT AMT'].fillna(0)
df['WITHDRAWAL AMT'] = df['WITHDRAWAL AMT'].fillna(0)

# Extract features for anomaly detection
X = df[['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']]

# Train an Isolation Forest model
model = IsolationForest(contamination='auto', random_state=42)
parameters = {'n_estimators': [50, 100, 1000], 'max_samples': ['auto', 100, 1000]}
grid_search = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
best_model = IsolationForest(contamination='auto', random_state=42, n_estimators=best_params['n_estimators'], max_samples=best_params['max_samples'])
best_model.fit(X)
# Extract unique account numbers
unique_account_numbers = list(df['Account No'].unique())
print("No of unique Accounts:   ", len(unique_account_numbers))

# Create a list to store instances of the AccountFeatures class
account_features_list = []

# Loop through unique account numbers
for account_number in unique_account_numbers:
    # Filter data for the current account number
    account_data = df[df['Account No'] == account_number]

    # Calculate existing average per month features
    avg_transactions_per_month = len(account_data) / len(account_data['DATE'].dt.to_period("M").unique())
    avg_withdrawals_per_month = account_data['WITHDRAWAL AMT'].count() / len(account_data['DATE'].dt.to_period("M").unique())
    avg_deposits_per_month = account_data['DEPOSIT AMT'].count() / len(account_data['DATE'].dt.to_period("M").unique())

    # Calculate average withdrawal amount and deposit amount
    avg_withdrawal_amount = account_data['WITHDRAWAL AMT'].mean()
    avg_deposit_amount = account_data['DEPOSIT AMT'].mean()

    # Calculate average balance amount (upper and lower)
    avg_balance_amount_upper = account_data['BALANCE AMT'].max()
    avg_balance_amount_lower = account_data['BALANCE AMT'].min()

    # Predict anomaly for the current account
    predictions = best_model.decision_function(account_data[['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']])
    anomaly_probability = 1 / (1 + (-predictions))
    overall_anomaly_score = anomaly_probability.mean()

    # Create an instance of the AccountFeatures class and append it to the list
    account_features = AccountFeatures(
        account_number,
        avg_transactions_per_month,
        avg_withdrawals_per_month,
        avg_deposits_per_month,
        avg_withdrawal_amount,
        avg_deposit_amount,
        avg_balance_amount_upper,
        avg_balance_amount_lower,
        anomaly_score=overall_anomaly_score
    )
    account_features_list.append(account_features)

# Print the account features
print("\nAccount Features:")
for account in account_features_list:
    print(f"Account No: {account.account_no}")
    print(f"Avg Transactions/Month: {account.avg_transactions_per_month}")
    print(f"Avg Withdrawals/Month: {account.avg_withdrawals_per_month}")
    print(f"Avg Deposits/Month: {account.avg_deposits_per_month}")
    print(f"Avg Withdrawal Amount: {account.avg_withdrawal_amount}")
    print(f"Avg Deposit Amount: {account.avg_deposit_amount}")
    print(f"Avg Balance Amount (Upper): {account.avg_balance_amount_upper}")
    print(f"Avg Balance Amount (Lower): {account.avg_balance_amount_lower}")
    # print(f"Overall Anomaly Score: {account.anomaly_score}")
    print("\n")

# Get user input for a new transaction
account_no = input("Enter Account No: ")
withdrawal_amt = float(input("Enter Withdrawal Amount: "))
deposit_amt = float(input("Enter Deposit Amount: "))
balance_amt = float(input("Enter Balance Amount:"))

# Create a DataFrame for the new transaction
new_transaction = pd.DataFrame({
    'WITHDRAWAL AMT': [withdrawal_amt],
    'DEPOSIT AMT': [deposit_amt],
    'BALANCE AMT': [balance_amt]
})

# Handling NaN values
new_transaction['DEPOSIT AMT'] = new_transaction['DEPOSIT AMT'].fillna(0)
new_transaction['WITHDRAWAL AMT'] = new_transaction['WITHDRAWAL AMT'].fillna(0)
# ... (previous code)

# Predict anomaly for the new transaction
predictions = best_model.decision_function(new_transaction[['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']])
anomaly_probability = 1 / (1 + (-predictions))  # Convert decision function values to probabilities
overall_anomaly_score = anomaly_probability.mean()

# Categorize the transaction based on the threshold
threshold = 0.5
anomaly_label = 1 if overall_anomaly_score > threshold else 0

# Print the result
print("\nNew Transaction Prediction:")
print(f"Account No: {account_no}")
print(f"Withdrawal Amount: {withdrawal_amt}")
print(f"Deposit Amount: {deposit_amt}")
print(f"Balance Amount: {balance_amt}")
print(f"Overall Anomaly Probability: {overall_anomaly_score}")
print(f"Anomaly Label: {anomaly_label}")

# Add anomaly probability to the new_transaction DataFrame
new_transaction['Anomaly Probability'] = anomaly_probability

# Embedding anomaly predictions for each feature
# Train the model again for each feature
for feature in ['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']:
    model.fit(df[[feature]])
    predictions = model.predict(df[[feature]])
    df[f'{feature}_anomaly'] = predictions

# Print anomaly predictions for each feature and transaction
for index, row in df.iterrows():
    overall_anomaly_score = (row['WITHDRAWAL AMT_anomaly'] + row['DEPOSIT AMT_anomaly'] + row['BALANCE AMT_anomaly']) / 3.0
    if overall_anomaly_score < 0:
        continue

# Sum up anomaly and non-anomaly transactions for unique account numbers
total_anomaly_counts = df.groupby('Account No').apply(lambda x: (x['WITHDRAWAL AMT_anomaly'] < 0).sum() +
                                                                  (x['DEPOSIT AMT_anomaly'] < 0).sum() +
                                                                  (x['BALANCE AMT_anomaly'] < 0).sum()).reset_index(name='Total Anomalies')

total_non_anomaly_counts = df.groupby('Account No').apply(lambda x: (x['WITHDRAWAL AMT_anomaly'] >= 0).sum() +
                                                                      (x['DEPOSIT AMT_anomaly'] >= 0).sum() +
                                                                      (x['BALANCE AMT_anomaly'] >= 0).sum()).reset_index(name='Total Non-Anomalies')

# Calculate averages and print the results
total_counts = pd.merge(total_anomaly_counts, total_non_anomaly_counts, on='Account No')
total_counts['Anomaly Percentage'] = (total_counts['Total Anomalies'] / (total_counts['Total Anomalies'] + total_counts['Total Non-Anomalies'])) * 100
total_counts['Non-Anomaly Percentage'] = (total_counts['Total Non-Anomalies'] / (total_counts['Total Anomalies'] + total_counts['Total Non-Anomalies'])) * 100

print("\nAccount Anomaly and Non-Anomaly Percentages:")
print(total_counts)
