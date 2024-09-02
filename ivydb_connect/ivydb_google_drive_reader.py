import io
import zipfile
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pandas as pd
import numpy as np
from ivydb_utils import IvyDBUtils
import tqdm

class IvyDBGoogleDriveReader:
    def __init__(self, credentials_file: str, folder_id: str, definitions_file: str):
        self.folder_id = folder_id
        self.ivy_db = IvyDBUtils(definitions_file)
        
        # Set up Google Drive service
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.service = build('drive', 'v3', credentials=credentials)

    def search_file(self, file_name: str) -> str:
        query = f"name='{file_name}' and '{self.folder_id}' in parents"
        results = self.service.files().list(
            q=query, spaces='drive', fields='files(id, name)'
        ).execute()
        items = results.get('files', [])
        
        if not items:
            raise FileNotFoundError(f"File {file_name} not found in the specified folder.")
        
        return items[0]['id']

    def download_and_extract(self, file_id: str) -> io.BytesIO:
        request = self.service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        file.seek(0)
        with zipfile.ZipFile(file) as zip_ref:
            txt_file = [f for f in zip_ref.namelist() if f.endswith('.txt')][0]
            print(f"Download {int(status.progress() * 100)}%.")
            return io.BytesIO(zip_ref.read(txt_file))

    def read_file(self, file_name: str, table_name: str) -> pd.DataFrame:
        file_id = self.search_file(file_name)
        file_content = self.download_and_extract(file_id)
        
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

# Example usage
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
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
