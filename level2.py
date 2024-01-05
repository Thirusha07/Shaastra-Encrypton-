import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import math
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

    # Check if the number of transactions is greater than the average transactions per month for the account
  
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
        anomaly_score=overall_anomaly_score  - 0.15
    )
    account_features_list.append(account_features)

# Print the account features
print("\nAccount Features:")
def View_Process(ano) :
    dt = {}
    for account in account_features_list:
        if account.account_no == ano : 
            dt["Avg Transactions/Month"]: account.avg_transactions_per_month
            dt["Avg Withdrawals/Month"]: account.avg_withdrawals_per_month
            dt["Avg Deposits/Month"]: account.avg_deposits_per_month
            dt["Avg Withdrawal Amount"] : account.avg_withdrawal_amount
            dt["Avg Deposit Amount"] : account.avg_deposit_amount
            dt["Avg Balance Amount (Upper)"] : account.avg_balance_amount_upper
            dt["Avg Balance Amount (Lower)"] : account.avg_balance_amount_lower
            dt["Overall Anomaly Score"] : account.anomaly_score
    return dt
       # print("PRE-PROCESSING DATA-SET WITH FEATURE EXTRACTION OVER HERE")
    

for account in account_features_list:
    print(f"Account No: {account.account_no}")
    print(f"Avg Transactions/Month: {account.avg_transactions_per_month}")
    print(f"Avg Withdrawals/Month: {account.avg_withdrawals_per_month}")
    print(f"Avg Deposits/Month: {account.avg_deposits_per_month}")
    print(f"Avg Withdrawal Amount: {account.avg_withdrawal_amount}")
    print(f"Avg Deposit Amount: {account.avg_deposit_amount}")
    print(f"Avg Balance Amount (Upper): {account.avg_balance_amount_upper}")
    print(f"Avg Balance Amount (Lower): {account.avg_balance_amount_lower}")
    print(f"Overall Anomaly Score: {account.anomaly_score}")
    print("\n")

print("PRE-PROCESSING DATA-SET WITH FEATURE EXTRACTION OVER HERE")

print("INPUT NEW TRANSACTION FOR ANOMALY DETECTION 2ND LAYER INTERFACE INPUT")
# Get user input for a new transaction
account_no = input("Enter Account No: ")
num_transactions = int(input("Enter the number of new transactions: "))
new_transactions_list = []

for _ in range(num_transactions):
    date = input("Enter Date (YYYY-MM-DD): ")
    withdrawal_amt = float(input("Enter Withdrawal Amount: "))
    deposit_amt = float(input("Enter Deposit Amount: "))
    balance_amt = float(input("Enter Balance Amount: "))

    new_transaction = {
        'Account No': account_no,
        'DATE': pd.to_datetime(date),
        'WITHDRAWAL AMT': withdrawal_amt,
        'DEPOSIT AMT': deposit_amt,
        'BALANCE AMT': balance_amt
    }

    new_transactions_list.append(new_transaction)

new_transactions_df = pd.DataFrame(new_transactions_list)

new_transactions_df['DEPOSIT AMT'] = new_transactions_df['DEPOSIT AMT'].fillna(0)
new_transactions_df['WITHDRAWAL AMT'] = new_transactions_df['WITHDRAWAL AMT'].fillna(0)

new_transactions_df['Month'] = new_transactions_df['DATE'].dt.to_period("M")
avg_transactions_per_month_new = new_transactions_df.groupby('Month').size().mean()

# Predict transactions as anomalies if the number of transactions is greater than the average transactions per month
minn = math.floor(math.ceil(avg_transactions_per_month) - 0.25*math.ceil(avg_transactions_per_month))
maxx = math.ceil(math.ceil(avg_transactions_per_month) + 0.25*math.ceil(avg_transactions_per_month))

if minn <= avg_transactions_per_month and maxx >= avg_transactions_per_month :
    print(f"\n\n Anomaly pattern-  Number of transactions is not within the range.\n\n\n")


# Predict anomaly for the new transactions
predictions_new = best_model.decision_function(new_transactions_df[['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']])
anomaly_probability_new = 1 / (1 + (-predictions_new))
overall_anomaly_score_new = anomaly_probability_new.mean()
if overall_anomaly_score_new > 0.3 :
    overall_anomaly_score_new -= 0.15
# Categorize the new transactions based on the threshold
threshold = 0.5
anomaly_label_new = 1 if overall_anomaly_score_new > threshold else 0

# ...

# ...

## ...

print("INPUT TRANSACTIONS ANALYSIS ")

# Print the result for the new transactions
print("\nNew Transaction Prediction:")
for idx, row in new_transactions_df.iterrows():
    print(f"Account No: {row['Account No']}")
    print(f"Date: {row['DATE']}")
    print(f"Withdrawal Amount: {row['WITHDRAWAL AMT']}")
    print(f"Deposit Amount: {row['DEPOSIT AMT']}")
    print(f"Balance Amount: {row['BALANCE AMT']}")
    print(f"Overall Anomaly Probability: {overall_anomaly_score_new}")
    print(f"Anomaly Label: {anomaly_label_new}")

    # Calculate the average amounts for the current account
    current_account_data = df[df['Account No'] == row['Account No']]
    avg_withdrawal_amount_current = current_account_data['WITHDRAWAL AMT'].mean()
    avg_deposit_amount_current = current_account_data['DEPOSIT AMT'].mean()
    avg_balance_amount_upper_current = current_account_data['BALANCE AMT'].max()
    avg_balance_amount_lower_current = current_account_data['BALANCE AMT'].min()

    # Check the reasons for anomaly
    if row['WITHDRAWAL AMT']!=0  and row['WITHDRAWAL AMT'] < (0.75 * avg_withdrawal_amount_current) or row['WITHDRAWAL AMT'] > (1.25 * avg_withdrawal_amount_current):
        print("Reason: Anomaly in Withdrawal-amount pattern behavior")
    elif row['DEPOSIT AMT'] !=0 and  row['DEPOSIT AMT'] < (0.75 * avg_deposit_amount_current) or row['DEPOSIT AMT'] > (1.25 * avg_deposit_amount_current):
        print("Reason: Anomaly in Deposit-amount pattern behavior")
    elif row['BALANCE AMT'] < (0.75 * avg_balance_amount_lower_current) or row['BALANCE AMT'] > (1.25 * avg_balance_amount_upper_current):
        print("Reason: Anomaly in Balance-amount pattern behavior")
    else:
        print("Reason: No anomaly detected")

    print("\n")

# ...



   


print("DATA-SET ANOMALIES VS NON-ANOMALIES ANALYSIS")
# Add anomaly probability to the new_transactions_df DataFrame
new_transactions_df['Anomaly Probability'] = anomaly_probability_new

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
