import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime

class EDA:
    def __init__(self, accepted_data_path, rejected_data_path):
        self.used_cols = ['loan_amnt','zip_code', 'loan_status', 'funded_amnt', 'funded_amnt_inv']
        self.acc_df = pd.read_csv(accepted_data_path, usecols=None)
        #self.rej_df = pd.read_csv(rejected_data_path, usecols=['Zip Code'])

    def data_info(self):
        self.acc_df.info()

    def raw_clean_up(self):
        colsToDrop = ["id", "member_id", "funded_amnt", "emp_title", "pymnt_plan", "url", "desc", "title", "zip_code", "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record", "revol_bal", "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "collections_12_mths_ex_med", "policy_code", "acc_now_delinq", "chargeoff_within_12_mths", "delinq_amnt", "tax_liens", "application_type", "pub_rec_bankruptcies", "addr_state",]
        # drop cols irrelevant to credit default analysis
        self.acc_df.drop(colsToDrop, axis=1, inplace=True)
        # drop cols that have more than 10% values as na
        self.acc_df = self.acc_df.dropna(axis=1, thresh=self.acc_df.shape[0]//10)
        self.acc_df = self.acc_df.iloc[:,0:23]
        # drop rows that are missing important values
        self.acc_df.dropna(axis=0, subset=["emp_length"], inplace=True)
        self.acc_df.dropna(axis=0, subset=["revol_util"], inplace=True)

    def refined_clean_up(self):
        # remove text data from term feature and store as numerical
        self.acc_df["term"] = pd.to_numeric(self.acc_df["term"].apply(lambda x:x.split()[0]))
        # remove the rows with loan_status as "Current"
        self.acc_df = self.acc_df[self.acc_df["loan_status"].apply(lambda x:False if x == "Current" else True)]

        # update loan_status as Fully Paid to 0 and Charged Off to 1
        self.acc_df["loan_status"] = self.acc_df["loan_status"].apply(lambda x: 0 if x == "Fully Paid" else 1)
        # update emp_length feature with continuous values as int
        # where (< 1 year) is assumed as 0 and 10+ years is assumed as 10 and rest are stored as their magnitude
        self.acc_df["emp_length"] = pd.to_numeric(self.acc_df["emp_length"].apply(lambda x:0 if "<" in x else (x.split('+')[0] if "+" in x else x.split()[0])))
        # look through the purpose value counts
        loan_purpose_values = self.acc_df["purpose"].value_counts()*100/self.acc_df.shape[0]

        # for annual_inc, the highest value is 6000000 where 75% quantile value is 83000, and is 100 times the mean
        # we need to remomve outliers from annual_inc i.e. 99 to 100%
        annual_inc_q = self.acc_df["annual_inc"].quantile(0.99)
        self.acc_df = self.acc_df[self.acc_df["annual_inc"] < annual_inc_q]
        # for open_acc, the highest value is 44 where 75% quantile value is 12, and is 5 times the mean
        # we need to remomve outliers from open_acc i.e. 99.9 to 100%
        open_acc_q = self.acc_df["open_acc"].quantile(0.999)
        self.acc_df = self.acc_df[self.acc_df["open_acc"] < open_acc_q]
        # for total_acc, the highest value is 90 where 75% quantile value is 29, and is 4 times the mean
        # we need to remomve outliers from total_acc i.e. 98 to 100%
        total_acc_q = self.acc_df["total_acc"].quantile(0.98)
        self.acc_df = self.acc_df[self.acc_df["total_acc"] < total_acc_q]
        # for pub_rec, the highest value is 4 where 75% quantile value is 0, and is 4 times the mean
        # we need to remomve outliers from pub_rec i.e. 99.5 to 100%
        pub_rec_q = self.acc_df["pub_rec"].quantile(0.995)
        self.acc_df = self.acc_df[self.acc_df["pub_rec"] <= pub_rec_q]

        # remove rows with less than 1% of value counts in paricular purpose 
        loan_purpose_delete = loan_purpose_values[loan_purpose_values<1].index.values
        self.acc_df = self.acc_df[[False if p in loan_purpose_delete else True for p in self.acc_df["purpose"]]]

        # extracting month and year from issue_date
        self.acc_df['month'] = self.acc_df['issue_d'].apply(lambda x: x.split('-')[0])
        self.acc_df['year'] = self.acc_df['issue_d'].apply(lambda x: x.split('-')[1])
        # get year from issue_d and replace the same
        self.acc_df["earliest_cr_line"] = pd.to_numeric(self.acc_df["earliest_cr_line"].apply(lambda x:x.split('-')[1]))

        # create bins for loan_amnt range
        bins = [0, 5000, 10000, 15000, 20000, 25000, 36000]
        bucket_l = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000','25000+']
        self.acc_df['loan_amnt_range'] = pd.cut(self.acc_df['loan_amnt'], bins, labels=bucket_l)
        # create bins for int_rate range
        bins = [0, 7.5, 10, 12.5, 15, 100]
        bucket_l = ['0-7.5', '7.5-10', '10-12.5', '12.5-15', '15+']
        self.acc_df['int_rate_range'] = pd.cut(self.acc_df['int_rate'], bins, labels=bucket_l)
        # create bins for annual_inc range
        bins = [0, 25000, 50000, 75000, 100000, 1000000]
        bucket_l = ['0-25000', '25000-50000', '50000-75000', '75000-100000', '100000+']
        self.acc_df['annual_inc_range'] = pd.cut(self.acc_df['annual_inc'], bins, labels=bucket_l)

        # create bins for installment range
        self.acc_df['installment'] = self.acc_df['installment'].apply(lambda x: self._installment(x))
        # create bins for dti range
        bins = [-1, 5.00, 10.00, 15.00, 20.00, 25.00, 50.00]
        bucket_l = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25%+']
        self.acc_df['dti_range'] = pd.cut(self.acc_df['dti'], bins, labels=bucket_l)

    def _standerdisedate(self, date):
        year = date.split("-")[0]
        if(len(year) == 1):
            date = "0"+date
        return date

    def _installment(self, n):
        if n <= 200:
            return 'low'
        elif n > 200 and n <=500:
            return 'medium'
        elif n > 500 and n <=800:
            return 'high'
        else:
            return 'very high'

    def loan_defaults(self):
        self.chargedOff = self.acc_df.loc[self.acc_df['loan_status'] == "Charged Off"]
        self.currentLoans = self.acc_df.loc[self.acc_df['loan_status'] == "Current"]
        self.fullyPaid = self.acc_df.loc[self.acc_df['loan_status'] == "Fully Paid"]
        loan_numbers = [self.chargedOff.shape[0], self.currentLoans.shape[0], self.fullyPaid.shape[0]]
        loan_numbers_labels = "Charged Off","Current Loans","Fully Paid"
        plt.pie(loan_numbers, labels=loan_numbers_labels, autopct='%1.1f%%')
        plt.title("Loan Status Aggregate by Number")
        plt.axis('equal')
        plt.legend(loan_numbers,title="Number of Loans",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.show()
    
    def loan_defaults_by_amount(self):
        data  = [{"Charged Off": self.chargedOff["funded_amnt_inv"].sum(), "Fully Paid": self.fullyPaid["funded_amnt_inv"].sum(), "Current": self.currentLoans["funded_amnt_inv"].sum()}]
        investment_sum = pd.DataFrame(data)
        chargedOffTotalSum = float(investment_sum["Charged Off"])
        fullyPaidTotalSum = float(investment_sum["Fully Paid"])
        currentTotalSum = float(investment_sum["Current"])
        self.loan_status = [chargedOffTotalSum,fullyPaidTotalSum,currentTotalSum]
        loan_status_labels = 'Charged Off','Fully Paid','Current'
        plt.pie(self.loan_status,labels=loan_status_labels,autopct='%1.1f%%')
        plt.title('Loan Status Aggregate Information')
        plt.axis('equal')
        plt.legend(self.loan_status,title="Loan Amount",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.show()


    def loan_purpose(self):
        # plotting pie chart for different types of purpose
        loans_purpose = self.acc_df.groupby(['purpose'])['funded_amnt_inv'].sum().reset_index()
        plt.figure(figsize=(14, 10))
        plt.pie(loans_purpose["funded_amnt_inv"],labels=loans_purpose["purpose"],autopct='%1.1f%%')
        plt.title('Loan purpose Aggregate Information')
        plt.axis('equal')
        plt.legend(self.loan_status,title="Loan purpose",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
        plt.show()


    def ratio(self, feature, figsize=(10,5), rsorted=True):
        plt.figure(figsize=figsize)
        if rsorted:
            dimension = sorted(self.acc_df[feature].unique())
        else:
            dimension = self.acc_df[feature].unique()
        values = []
        for f in dimension:
            filter = self.acc_df[self.acc_df[feature]==f]
            count = len(filter[filter["loan_status"]==1])
            values.append(count*100/filter["loan_status"].count())
        plt.bar(dimension, values)
        plt.title("Loan Defaults versus "+str(feature))
        plt.xlabel(feature, fontsize=16)
        plt.ylabel("default %", fontsize=16)
        plt.show()

    def bar(self, x, figsize=(10,5)):
        plt.figure(figsize=figsize)
        sns.barplot(x=x, y='loan_status', data=self.acc_df.sort_values(x))
        plt.title("Loan Defaults versus "+str(x))
        plt.xlabel(x, fontsize=16)
        plt.ylabel("default ratio", fontsize=16)
        plt.show()