import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_and_clean_data(file_path):
  
    # Loads, cleans, and prepares data for RFM analysis and outlier removal. Args:file_path (str): The path to the CSV file. Returns: pandas.DataFrame: The cleaned and processed DataFrame.

    # Load data
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

    # Convert CustomerID to string and create Amount column
    # retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # Compute RFM metrics
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']

    # Corrected InvoiceDate format
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], dayfirst=True, errors='coerce')


    max_date = retail['InvoiceDate'].max()

    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days

    rfm = pd.merge(rfm_m, rfm_f, on="CustomerID", how="inner")
    rfm = pd.merge(rfm, rfm_p, on="CustomerID", how="inner")
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    # print("size:",rfm.shape)
    # Remove outliers
    Q1 = rfm.quantile(0.05)
    Q3 = rfm.quantile(0.95)
    IQR = Q3 - Q1

    rfm = rfm[(rfm.Amount >= Q1[0] - 1.5 * IQR[0]) & (rfm.Amount <= Q3[0] + 1.5 * IQR[0])]
    rfm = rfm[(rfm.Recency >= Q1[2] - 1.5 * IQR[2]) & (rfm.Recency <= Q3[2] + 1.5 * IQR[2])]
    rfm = rfm[(rfm.Frequency >= Q1[1] - 1.5 * IQR[1]) & (rfm.Frequency <= Q3[1] + 1.5 * IQR[1])]
    
    
    return rfm
   


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)

    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

    # Instantiate
    scaler = StandardScaler()

    # fit_transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)

    # rfm_df_scaled
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    # Added by Dhruv
    rfm_output_path= os.path.join(os.path.dirname(__file__),'static','rfm_result.csv')
    rfm.to_csv(rfm_output_path,index=False)
    rfm_df_scaled_output_path= os.path.join(os.path.dirname(__file__),'static','rfm_scaled_result.csv')
    rfm_df_scaled.to_csv(rfm_df_scaled_output_path,index=False)
    
    return rfm, rfm_df_scaled