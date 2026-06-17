import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    r"C:\Users\hasin\OneDrive\Desktop\CODSOFT\Credit_Card_Fraud_Detection\creditcard.csv"
)

data['Class'].value_counts().plot(kind='bar')

plt.title("Credit Card Fraud Detection")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.show()