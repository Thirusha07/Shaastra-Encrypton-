import pandas as pd
class AccountFeatures:
    def __init__(self, account_no,
                 avg_transactions_per_month, avg_withdrawals_per_month, avg_deposits_per_month,
                 avg_withdrawal_amount, avg_deposit_amount,
                 avg_balance_amount_upper, avg_balance_amount_lower):
        self.account_no = account_no
        self.avg_transactions_per_month = avg_transactions_per_month
        self.avg_withdrawals_per_month = avg_withdrawals_per_month
        self.avg_deposits_per_month = avg_deposits_per_month
        self.avg_withdrawal_amount = avg_withdrawal_amount
        self.avg_deposit_amount = avg_deposit_amount
        self.avg_balance_amount_upper = avg_balance_amount_upper
        self.avg_balance_amount_lower = avg_balance_amount_lower

# Replace 'your_excel_file.xlsx' with the actual path to your Excel file
excel_file_path = 'iit_data.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

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

    # Create an instance of the AccountFeatures class and append it to the list
    account_features = AccountFeatures(
        account_number,
        avg_transactions_per_month,
        avg_withdrawals_per_month,
        avg_deposits_per_month,
        avg_withdrawal_amount,
        avg_deposit_amount,
        avg_balance_amount_upper,
        avg_balance_amount_lower
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
    print("\n")
def View_Process(ano) :
    dt = {}
    for account in account_features_list:
        if account.account_no == ano : 
            dt["Average Transactions per Month"] = account.avg_transactions_per_month
            dt["Average Withdrawals per Month"] = account.avg_withdrawals_per_month
            dt["Average Deposits per Month"] = account.avg_deposits_per_month
            dt["Average Withdrawal Amount"]  = account.avg_withdrawal_amount
            dt["Average Deposit Amount"]  = account.avg_deposit_amount
            dt["Average Balance Amount (MAX)"]  = account.avg_balance_amount_upper
            dt["Average Balance Amount (MIN)"]= account.avg_balance_amount_lower
           
    return dt
