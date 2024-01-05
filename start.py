import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class AccountFeatures:
    def __init__(self, account_no,
                 avg_transactions_per_month, avg_withdrawals_per_month, avg_deposits_per_month,
                 avg_withdrawal_amount, avg_deposit_amount,
                 avg_balance_amount_upper, avg_balance_amount_lower, anomaly_prediction):
        self.account_no = account_no
        self.avg_transactions_per_month = avg_transactions_per_month
        self.avg_withdrawals_per_month = avg_withdrawals_per_month
        self.avg_deposits_per_month = avg_deposits_per_month
        self.avg_withdrawal_amount = avg_withdrawal_amount
        self.avg_deposit_amount = avg_deposit_amount
        self.avg_balance_amount_upper = avg_balance_amount_upper
        self.avg_balance_amount_lower = avg_balance_amount_lower
        self.anomaly_prediction = anomaly_prediction

# Replace 'your_excel_file.xlsx' with the actual path to your Excel file
excel_file_path = 'iit_data.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Handling NaN values in 'DEPOSIT AMT' and 'WITHDRAWAL AMT'
df['DEPOSIT AMT'] = df['DEPOSIT AMT'].fillna(0)
df['WITHDRAWAL AMT'] = df['WITHDRAWAL AMT'].fillna(0)

# Extract features for anomaly detection
X = df[['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']]

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train an Isolation Forest model
model = IsolationForest(contamination='auto', random_state=42)
parameters = {'n_estimators': [50, 100, 200], 'max_samples': ['auto', 100, 200]}
grid_search = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_scaled)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
best_model = IsolationForest(contamination='auto', random_state=42, n_estimators=best_params['n_estimators'], max_samples=best_params['max_samples'])
best_model.fit(X_scaled)

# Predict anomalies for the entire dataset
predictions = best_model.predict(X_scaled)

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

    # Get the anomaly prediction for the current account number
    anomaly_prediction = predictions[df['Account No'] == account_number]

    # Create an instance of the AccountFeatures class with anomaly prediction and append it to the list
    account_features = AccountFeatures(
        account_number,
        avg_transactions_per_month,
        avg_withdrawals_per_month,
        avg_deposits_per_month,
        avg_withdrawal_amount,
        avg_deposit_amount,
        avg_balance_amount_upper,
        avg_balance_amount_lower,
        anomaly_prediction
    )
    account_features_list.append(account_features)

# Print the account features with anomaly predictions
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
    print(f"Anomaly Prediction: {account.anomaly_prediction}")
    print("\n")
    print(list(account.anomaly_prediction).count(1))
    print(list(account.anomaly_prediction).count(-1))
    