import io
import os
import zipfile
import hashlib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pandas as pd
import numpy as np
from ivydb_utils import IvyDBUtils
from pathlib import Path
from numba import jit

import tqdm

class IvyDBGoogleDriveReader:
    def __init__(self, credentials_file: str, folder_id: str, definitions_file: str, cache_dir: str = 'cache'):
        self.folder_id = folder_id
        self.ivy_db = IvyDBUtils(definitions_file)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up Google Drive service
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.service = build('drive', 'v3', credentials=credentials)

    def search_file(self, file_name: str) -> str:
        query = f"name='{file_name}' and '{self.folder_id}' in parents"
        results = self.service.files().list(
            q=query, spaces='drive', fields='files(id, name, modifiedTime)'
        ).execute()
        items = results.get('files', [])
        
        if not items:
            raise FileNotFoundError(f"File {file_name} not found in the specified folder.")
        
        return items[0]

    def get_cache_path(self, file_name: str) -> Path:
        return self.cache_dir / file_name

    def is_cache_valid(self, file_name: str, modified_time: str) -> bool:
        cache_path = self.get_cache_path(file_name)
        if not cache_path.exists():
            return False
        
        cache_modified_time = cache_path.stat().st_mtime
        drive_modified_time = pd.to_datetime(modified_time).timestamp()
        
        return cache_modified_time >= drive_modified_time

    def download_and_cache(self, file_id: str, file_name: str) -> Path:
        request = self.service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        
        cache_path = self.get_cache_path(file_name)
        with open(cache_path, 'wb') as f:
            f.write(file.getvalue())
        
        return cache_path
    


    def download_and_extract(self, file_id: str) -> io.BytesIO:
        request = self.service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        
        file.seek(0)
        with zipfile.ZipFile(file) as zip_ref:
            txt_file = [f for f in zip_ref.namelist() if f.endswith('.txt')][0]
            return io.BytesIO(zip_ref.read(txt_file))

    def read_file(self, file_name: str, table_name: str) -> pd.DataFrame:
        file_info = self.search_file(file_name)
        print(f'file_info: {file_info}')
        #print(f'File ID: {file_info['id']}')
        cache_path = self.get_cache_path(file_name)
        if not self.is_cache_valid(file_name, file_info['modifiedTime']):
            cache_path = self.download_and_cache(file_info['id'], file_name)

        #file_content = self.download_and_extract(file_id)
        with zipfile.ZipFile(cache_path) as zip_ref:
            txt_file = [f for f in zip_ref.namelist() if f.endswith('.txt')][0]
            with zip_ref.open(txt_file) as file_content:
                dtype_dict = self.ivy_db.get_dtype_dict(table_name)
                float_dtype_dict = {col: 'float64' if dtype.startswith('int') else dtype for col, dtype in dtype_dict.items()}

                date_columns = [col['name'] for col in self.ivy_db.get_table_structure(table_name) if col['type'] == 'date']
                
                df = pd.read_csv(
                    file_content,
                    delimiter='\t',
                    names=self.ivy_db.get_column_names(table_name),
                    dtype=float_dtype_dict,
                    parse_dates=date_columns,
                    na_values=['-99.99', 'nan', 'NaN', '']  # Add any other strings that represent NaN in your data
                )

                for col, dtype in dtype_dict.items():
                    if dtype.startswith('int'):
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                
                # Handle missing values (assuming -99.99 is used for missing values)
                
        return df


    def clear_cache(self):
        for file in self.cache_dir.glob('*'):
            file.unlink()

# Example sage
def rank_strikes(group):
    atm = 1.0
    group['Strike_Diff'] = abs(group['Percent_ATM'] - atm)
    above_atm = group[group['Percent_ATM'] >= atm].sort_values('Strike_Diff')
    below_atm = group[group['Percent_ATM'] < atm].sort_values('Strike_Diff', ascending=False)
    
    above_atm['Rank'] = range(1, len(above_atm) + 1)
    below_atm['Rank'] = range(-len(below_atm), 0)
    
    return pd.concat([above_atm, below_atm])

@jit(nopython=True)
def linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope

if __name__ == "__main__":
    reader = IvyDBGoogleDriveReader(
        credentials_file='testproject1-419520-61c1efd44a96.json',
        folder_id='1AlYGA105eq8xdJQzJsSNWFBmIZvvEcsu',
        definitions_file='ivydb_table_definitions.json'
    )
    
    try:
        df = reader.read_file('IVYSECPRD_199601.zip', 'Security_Price')
        df2= reader.read_file('IVYOPPRCD_199601.zip', 'Option_Price')
        print(df.head())
        print(df.dtypes)
        print(df2.head())
        print(df2.dtypes)
        # Assuming security_price_df and option_price_df are your DataFrames

        # Create indexes
        df = df.set_index(['Security ID', 'Date'])
        df2 = df2.set_index(['Security ID', 'Date'])
        #df=df[df.index.get_level_values('Security ID')==108105]

        # Join the DataFrames
        joined_df = df[['Close Price', 'Adjustment Factor2']].join(
            df2, how='inner'
        )

        # Reset the index if you want 'Security ID' and 'Date' as columns
        joined_df = joined_df.reset_index()
        testfor=50000
        #joined_df=joined_df.iloc[:testfor]

        joined_df['Percent_ATM'] = (joined_df['Strike'] / 1000) / joined_df['Close Price']
        #print(joined_df.head())
        joined_df=joined_df.pivot_table(index=['Security ID', 'Date','Close Price', 'Adjustment Factor2', 'Expiration','Percent_ATM','Strike','Special Settlement'], columns='Call/Put').reset_index()



        @jit(nopython=True)
        def surf_params_numba(percent_atm, security_id, date, expiration,texp,iv,rank):
            n = len(percent_atm)
            slopes = np.zeros(n, dtype=np.float64)
            atm_iv = np.zeros(n, dtype=np.float64)

            # Create a unique identifier for each group
            group_ids = security_id * 1000000000000 + date * 1000000 + expiration
            unique_groups = np.unique(group_ids)
            
            for group in unique_groups:
                mask = group_ids == group
                mask=mask&(~np.isnan(iv))
                group_percent_atm = percent_atm[mask]
                group_ranks=rank[mask]
                group_iv=iv[mask]
                group_texp=texp[mask]
                log_strike=np.log(group_percent_atm)
                atm_vol=np.nan
                if(~np.isnan(group_iv).all()):
                    min_rank=np.min(np.abs(group_ranks))
                    atm_vol=group_iv[np.abs(group_ranks)==min_rank][0]
                    atm_strike=log_strike[np.abs(group_ranks)==min_rank][0]
                    log_strike=log_strike-atm_strike
                    #print(f'atm_vol: {atm_vol}')
                    #print(f'atm_strike: {atm_strike}')
                    #print(f'log_strike: {log_strike}')
                    #print(f'group_iv: {group_iv}')
                if(~np.isnan(atm_vol)):
                    if(len(group_iv)>1):

                        slopes[mask] = linear_regression(log_strike, group_iv-atm_vol)
                        slopes[mask]=slopes[mask]*np.sqrt(group_texp)/10
                        atm_vol=atm_vol-slopes[mask][0]*10*atm_strike/np.sqrt(group_texp[0])
                        atm_iv[mask]=atm_vol
            
            return atm_iv,slopes

        @jit(nopython=True)
        def rank_strikes_numba(percent_atm, security_id, date, expiration):
            n = len(percent_atm)
            ranks = np.zeros(n, dtype=np.int64)
            
            # Create a unique identifier for each group
            group_ids = security_id * 1000000000000 + date * 1000000 + expiration
            unique_groups = np.unique(group_ids)
            
            for group in unique_groups:
                mask = group_ids == group
                group_percent_atm = percent_atm[mask]
                group_ranks = np.zeros(len(group_percent_atm), dtype=np.int64)
                
                above = group_percent_atm >= 1.0
                below = ~above
                
                group_ranks[above] = np.argsort(np.abs(group_percent_atm[above] - 1.0)) #+ 1
                group_ranks[below] = -(np.argsort(np.abs(group_percent_atm[below] - 1.0))[::1] + 1)
                
                ranks[mask] = group_ranks
            
            return ranks


        def apply_rank_strikes_numba(df):
            # Convert date columns to integers (days since epoch)
            date_int = df['Date'].astype(int) // 10**9
            expiration_int = df['Expiration'].astype(int) // 10**9
            #texp=(df['Expiraton']-df['Date']).dt.days/365.25
            #print(f'texp: {texp}')
            expiration_dt=df['Expiration']
            date_dt=df['Date']
            texp=(expiration_dt-date_dt).dt.days/365.25
            
            # Apply the Numba function
            df['Rank'] = rank_strikes_numba(
                df['Percent_ATM'].astype(float).to_numpy(),
                df['Security ID'].astype(int).to_numpy(),
                date_int.to_numpy(),
                expiration_int.to_numpy()
            )
            #print(f'date_int: {date_int}')
            #print(f'expiration_int: {expiration_int}')
            #print(f'expiration_int-date_int: {(expiration_int.to_numpy()-date_int.to_numpy())/365.25}')
            atm_iv,slopes = surf_params_numba(
                df['Percent_ATM'].astype(float).to_numpy(),
                df['Security ID'].astype(int).to_numpy(),
                date_int.to_numpy(),
                expiration_int.to_numpy(),
                texp.to_numpy(),
                df['Delta Weighted IVol'].astype(float).to_numpy(),
                df['Rank'].astype(int).to_numpy()
            )
            df['ATM IV'] = atm_iv
            df['Slope'] = slopes
            return df

        # Use the function
        joined_df["Delta Weighted IVol"]=(1-joined_df['Delta']['C'])*joined_df['Implied Volatility']['C']+joined_df['Delta']['C']*joined_df['Implied Volatility']['P']
        joined_df = apply_rank_strikes_numba(joined_df)
        #joined_df = joined_df.groupby(['Security ID', 'Date','Expiration']).apply(rank_strikes).reset_index(drop=True)
        #joined_df = apply_rank_strikes_numba(joined_df)
        #input(joined_df['Delta']['C'])
        
        print(f'Joined DataFrame shape: {joined_df.shape}')
        print(joined_df.head())
        joined_df.to_csv('joined_df.csv', index=False)
        print(f'joined_df.columns: {joined_df.columns}')
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
