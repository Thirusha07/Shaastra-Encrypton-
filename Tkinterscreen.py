import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from pre_processed import * 

class AnomalyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Detection on Financial Transactions")

        # Account Number Entry
        self.account_label = tk.Label(root, text="Enter Account No:")
        self.account_label.pack()
        self.account_entry = tk.Entry(root)
        self.account_entry.pack()

        # Menu Options
        self.menu_label = tk.Label(root, text="Select an option:")
        self.menu_label.pack()

        options = ["About Model", "View Extraction-Features", "View Anomaly Percentages", "Input Transaction"]
        self.menu_var = tk.StringVar(root)
        self.menu_var.set(options[0])
        self.menu = tk.OptionMenu(root, self.menu_var, *options)
        self.menu.pack()

        # Button to trigger actions
        self.submit_button = tk.Button(root, text="Submit", command=self.perform_action)
        self.submit_button.pack()

    def perform_action(self):
        account_no = self.account_entry.get()
        selected_option = self.menu_var.get()

        if not account_no:
            messagebox.showerror("Error", "Please enter an Account No.")
            return

        if selected_option == "About Model":
            self.show_about_model()
        elif selected_option == "View Extraction-Features":
            self.view_extraction_features(account_no)
        elif selected_option == "View Anomaly Percentages":
            self.view_anomaly_percentages(account_no)
        elif selected_option == "Input Transaction":
            self.input_transaction(account_no)

    def show_about_model(self):
        about_model_text = """
        Problem-Statement

        The challenge of implementing and improving behavioral analytics solutions for detecting anomalies in financial transactions is crucial for enhancing security against fraudulent activities. This task involves leveraging the data and perform analysis to develop a system that can effectively identify suspicious (Anomalies) behavior within the vast volume of financial transactions.

        Data-set

        A suitable data-set that involves financial transaction data is used for training the model on distinguishing the anomalies. Here note that the data-set is not targeted (without target class . Since there does not exist an output class priorly specified in the transaction data: leads to the development of an unsupervised machine learning model. The Data set consists of 8 attributes out of which 6 attributes can be taken into consideration after pre-processing

        Approach

        The overview of the approach focuses on identifying anomalies and its reasoning patterns (cause-effect) by following a series of steps on achieving data collection, integration, training & tuning the model, feature engineering, behavioral profiling and authentication reaction set-up

        Pre-Processing

        The data-set is to be initially set to handle imbalances as the data is of large size approx 1.5L+ entries of transactions. Imbalances like Nan values ( For example: A transaction may be either of type withdrawal or deposit which may be lead to Nan values in the records

        Study-Analysis

        Analyze the nature of the data-set with respect to each account and observe any affecting factors that can be used as parameters / hyperparameters for our learning model based on Frequency and magnitude For example: Frequency -- Average No of withdrawal/month Magnitude --Average amount withdrawn

        Training model

        We have opted for the usage of isolated forest machine learning model (unsupervised) that follows ensemble learning and focuses on isolating the anomalies in the data-set that employs isolation and tracking of the anomalies in the subset (tree) of the data that ensures each tree captures different aspects of the overall pattern in the data. Feature influence and importance are also supported in this model that enables us to locate and identify the outliers (Anomalies) from the historical finance transaction data. Hyperparameters can be chosen and applied for tuning and optimizing the model’s performance

        Behaviour

        Create behavioral profiles for users and entities based on historical transaction data. Regularly update profiles to adapt to changes in user behavior and transaction patterns.

        Reaction-set

        Once the model is successfully trained and an incoming transaction is faced and detected as an anomaly, This can lead to the prompt of MFA authentication like delivering OTP to a mobile for proceeding with the transaction based on the strength of anomaly prediction. This can ensure the security and improvement of the transaction after identifying the anomalies in the transactions

        Further Enhancements

        In the course of crediting funds to an account, our predictive analysis assumes a pivotal role. Should an anomaly be identified during this process, an automated message is promptly dispatched to the intended recipient, elucidating the nature of the detected anomaly. Conversely, when the system determines the transaction as legitimate, it seamlessly proceeds with the necessary actions. In the event of suspicious activity, the system affords the option to freeze the associated account, mitigating the risk of potential security breaches and fortifying our defense against unauthorized access.
        """

        about_model_window = tk.Toplevel(self.root)
        about_model_window.title("About Model")

        about_model_text_widget = scrolledtext.ScrolledText(about_model_window, wrap=tk.WORD, width=80, height=30, font=('Helvetica', 10))
        about_model_text_widget.insert(tk.END, about_model_text)
        about_model_text_widget.config(state=tk.DISABLED)
        about_model_text_widget.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        
        
        
        messagebox.showinfo("About Model", "This is a simplified example of an anomaly detection model for financial transactions.")
    def view_extraction_features(self, account_no):
        # Call View_process from level2 module
        result = View_Process(account_no)

        # Display the result on the screen
        result_window = tk.Toplevel(self.root)
        result_window.title("Extraction Features Result")

        result_label = tk.Label(result_window, text=f"Result for Account No {account_no}:", font=('Helvetica', 12, 'bold'))
        result_label.grid(row=0, column=0, sticky=tk.W)

        # Iterate through the dictionary items and display them on separate lines
        row_num = 1
        for key, value in result.items():
            label_text = f"{key}: {value}"
            label = tk.Label(result_window, text=label_text, font=('Helvetica', 10))
            label.grid(row=row_num, column=0, sticky=tk.W)
            row_num += 1

        # Adjusting row and column weights for better appearance
        result_window.grid_rowconfigure(1, weight=1)
        result_window.grid_columnconfigure(0, weight=1)

    def view_anomaly_percentages(self, account_no):
        # Placeholder function
        messagebox.showinfo("Anomaly Percentages", f"Anomaly percentages for Account No {account_no} will be displayed here.")

    def input_transaction(self, account_no):
        input_window = tk.Toplevel(self.root)
        input_window.title(f"Input Transaction for Account No {account_no}")

        # Labels and Entry widgets for withdraw, deposit, and balance
        withdraw_label = tk.Label(input_window, text="Withdraw:")
        withdraw_label.grid(row=0, column=0, padx=10, pady=10)
        withdraw_entry = tk.Entry(input_window)
        withdraw_entry.grid(row=0, column=1, padx=10, pady=10)

        deposit_label = tk.Label(input_window, text="Deposit:")
        deposit_label.grid(row=1, column=0, padx=10, pady=10)
        deposit_entry = tk.Entry(input_window)
        deposit_entry.grid(row=1, column=1, padx=10, pady=10)

        balance_label = tk.Label(input_window, text="Balance:")
        balance_label.grid(row=2, column=0, padx=10, pady=10)
        balance_entry = tk.Entry(input_window)
        balance_entry.grid(row=2, column=1, padx=10, pady=10)

        submit_button = tk.Button(input_window, text="Submit", command=lambda: self.process_transaction(account_no, withdraw_entry.get(), deposit_entry.get(), balance_entry.get()))
        submit_button.grid(row=3, column=0, columnspan=2, pady=10)

    def process_transaction(self, account_no, withdraw, deposit, balance):
        # Placeholder predict function, replace with your actual prediction logic
        prediction_result, behavior = self.predict(account_no, withdraw, deposit, balance)

        result_window = tk.Toplevel(self.root)
        result_window.title("Transaction Prediction Result")

        result_label = tk.Label(result_window, text=f"Result for Account No {account_no}:\nPrediction: {prediction_result}\nBehavior: {behavior}", font=('Helvetica', 12))
        result_label.grid(row=0, column=0, padx=10, pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectionApp(root)
    root.mainloop()
