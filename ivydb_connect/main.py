import os
import pytz
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from dateutil.relativedelta import relativedelta

class OptAlpha():
    
    def __init__(self,instruments,trade_range,dfs):
        self.instruments = instruments 
        self.trade_range = trade_range
        self.dfs = dfs

    def archive_constructor(self,dt):
        if dt.year >= 1990 and dt.year <= 2021:
            return "ALLSPX_1990_2021.zip"
        elif dt.year == 2022:
            return "ALLSPX_2022.zip"
        elif dt.year == 2023:
            return "ALLSPX_2023.zip"

    def filename_constructor(self,dt):
        return f"SPX_{dt.year}.csv"

    def screen_universe(self,df,universe):
        df = df.loc[np.where(np.logical_or(df.volume == 0, df.openinterest == 0),False,True)]
        df.expiration = pd.to_datetime(df.expiration).dt.tz_localize("UTC")
        df.quotedate = pd.to_datetime(df.quotedate).dt.tz_localize("UTC")
        df["dte"] = (df.expiration - df.quotedate).apply(lambda x: x.days)
        df = df.drop(columns=[" exchange","optionext","optionalias","bid","ask","theta","vega","gamma","IVBid","IVAsk"])
        return df.set_index("optionroot",drop=True)

    def load_buffer(self,load_from,min_buffer_len=100,min_hist_len=2): 
        #specific to data source
        #specification for this function:: have a buffer data structure of historical data around variable load_from
        #data available from 1990-2023
        '''
        ALLSPX_1990_2021.zip for SPX_1990.csv to SPX_2021.csv
        ALLSPX_2022.zip for SPX_2022.csv
        ALLSPX_2023.zip for SPX_2023.csv
        '''
        dir = "/Users/admin/Desktop/spxopt/"
        if any(dt >= load_from for dt in self.data_buffer_idx):
            return 
        #[]
        #[a,b,c,d,e] e < load_from
        #[d,e]
        self.data_buffer = self.data_buffer[-min_hist_len:]
        self.data_buffer_idx = self.data_buffer_idx[-min_hist_len:]
        while len(self.data_buffer) < min_buffer_len:
            while self.filename_constructor(dt=load_from) in self.loaded:
                load_from += relativedelta(days=1)
            datfile = self.filename_constructor(dt=load_from)
            if datfile not in self.unzipped:
                an = self.archive_constructor(dt=load_from)
                with zipfile.ZipFile(dir+an,"r") as archive:
                    archive.extractall(path=dir+"pyrun/")
                    all_files=archive.namelist()
                    self.unzipped = self.unzipped.union(set(all_files))
            optdat = pd.read_csv(dir+"pyrun/"+datfile)
            optdat = self.screen_universe(df=optdat,universe=self.instruments)
            for date in sorted(set(optdat.quotedate)):
                self.data_buffer.append(optdat.loc[optdat.quotedate == date])
                self.data_buffer_idx.append(date.to_pydatetime())
            os.remove(dir+"pyrun/"+datfile)
            self.loaded.add(datfile)
        self.compute_buffer()

    def compute_buffer(self):
        from copy import deepcopy
        strat_buffer = []
        for optdat,optidx in zip(self.data_buffer,self.data_buffer_idx):
            data = deepcopy(optdat)
            data = data.loc[data.dte > 7] 
            data = data.loc[data.dte == np.min(data.dte)] #vol surface > (term structure, smile) T,K
            data["strike_dist"] = np.abs(data.underlying_last - data.strike)
            data = data.loc[data.strike_dist == np.min(data.strike_dist)] #1400 (1450) 1500
            data = data.loc[data.strike == np.min(data.strike)]
            strat_buffer.append(data)
        self.strat_buffer = strat_buffer

    def get_pnl(self,date,last):
        if date not in self.data_buffer_idx:
            return 0.0
        cur_idx = self.data_buffer_idx.index(date)
        curr = self.data_buffer[cur_idx]
        prev= self.data_buffer[cur_idx-1]
        pnl=0.0
        for ticker,positions in last.items():
            for call,unit in zip(positions["C"],positions["CU"]):
                pricedelta = curr.at[call,"last"]-prev.at[call,"last"] if call in curr.index and call in prev.index else 0.0
                pnl += pricedelta * unit
            for put,unit in zip(positions["P"],positions["PU"]):
                pricedelta = curr.at[put,"last"]-prev.at[put,"last"] if put in curr.index and put in prev.index else 0.0
                pnl += pricedelta * unit
        return pnl
    
    def _default_pos(self):
        return defaultdict(lambda : {"S":0, "C":[],"P":[],"CU":[],"PU":[]})
    
    def compute_signals(self,date,capital):
        if date not in self.data_buffer_idx:
            return 
        date_data = self.strat_buffer[self.data_buffer_idx.index(date)]
        notional_leverage = 3
        notional_per_trade = capital * notional_leverage
        signal_dict = self._default_pos()
        for inst in self.instruments:
            pos = notional_per_trade / date_data.underlying_last.values[0] *-1
            if len(date_data) == 2:
                signal_dict[inst] = {
                    "S": 0,
                    "C": [date_data.loc[date_data.type == "call"].index.values[0]],
                    "P": [date_data.loc[date_data.type == "put"].index.values[0]],
                    "CU": [pos],
                    "PU": [pos],
                }
        return signal_dict

    async def run_simulation(self):
        trade_start = self.trade_range[0]
        trade_end = self.trade_range[1]
        trade_range = pd.date_range(
            start=datetime(trade_start.year,trade_start.month,trade_start.day),
            end=datetime(trade_end.year,trade_end.month,trade_end.day),
            freq="D",
            tz=pytz.utc
        )

        portfolio_df = pd.DataFrame(index=trade_range).reset_index().rename(columns={"index":"datetime"})
        portfolio_df.at[0,"capital"] = 10000.0

        self.data_buffer = []
        self.data_buffer_idx = []
        self.unzipped = set()
        self.loaded = set()
        last_positions=self._default_pos()
        for i in portfolio_df.index:
            date = portfolio_df.at[i,"datetime"]
            self.load_buffer(load_from=date,min_buffer_len=180,min_hist_len=2)
            
            if i != 0:
                day_pnl = self.get_pnl(date=date,last=last_positions)
                portfolio_df.at[i,"capital"] = portfolio_df.at[i-1,"capital"]+day_pnl
            
            signal_dict = self.compute_signals(date=date,capital=portfolio_df.at[i,"capital"])
            signal_dict = signal_dict if signal_dict else last_positions

            last_positions = signal_dict
            if i%20 == 0: print(portfolio_df.at[i,"capital"])
        
        return portfolio_df

from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload

#async 
def main():
    trade_start = datetime(2000,1,1,tzinfo=pytz.utc)
    trade_end = datetime(2023,1,1,tzinfo=pytz.utc)
    #strategy of going short a ATM straddle on SPX
    #strat = OptAlpha(
    #    instruments=["SPX"],
    #    trade_range=(trade_start,trade_end),
    #    dfs={}
    #)import gdown

    # Replace with your file ID

    # Path to the credentials file
    SERVICE_ACCOUNT_FILE = 'testproject1-419520-61c1efd44a96.json'

    # Scopes required to access Google Drive
    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Authenticate and create the service
    credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    folder_id="1AlYGA105eq8xdJQzJsSNWFBmIZvvEcsu"
    # Function to search for a file by name
    def search_file(file_name, folder_id=None, shared_drive=False):
        # Base query for searching by file name
        query = f"name='{file_name}'"

        # If searching in a specific folder, modify the query
        if folder_id:
            query += f" and '{folder_id}' in parents"

        # If searching in a shared drive, include the correct parameters
        if shared_drive:
            results = service.files().list(
                q=query,
                spaces='drive',
                corpora='drive',  # Search within shared drives
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields='files(id, name)').execute()
        else:
            results = service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)').execute()

        items = results.get('files', [])

        if not items:
            print('No files found.')
            return None
        else:
            for item in items:
                print(f"Found file: {item['name']} (ID: {item['id']})")
                return item['id']
    # Example: Search for a file named "example.csv"
    def list_drives():
        results = service.drives().list(fields="drives(id, name)").execute()
        drives = results.get('drives', [])

        if not drives:
            print('No drives found.')
        else:
            for drive in drives:
                print(f"Drive name: {drive['name']} (ID: {drive['id']}) - Shared Drive")

# Call the function
    list_drives()
    # Function to load headers from a separate file
    def load_headers(header_file):
        with open(header_file, 'r') as file:
            headers = file.readline().strip().split(',')
        return headers

    file_id = search_file("IVYSECPRD_199601.zip",folder_id=folder_id)

    # Download the file if found
    if file_id:
        request = service.files().get_media(fileId=file_id)
        download_path = 'downloaded_example.zip'
        fh = io.FileIO(download_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            # Assuming there's only one .txt file in the zip
            csv_filename = [name for name in zip_ref.namelist() if name.endswith('.txt')][0]
            print(f"Extracting and reading {csv_filename} from the zip file.")
            
            headers = load_headers('IVYSECPRD_headers.txt')  # Update this path based on your file type

            with zip_ref.open(csv_filename) as csv_file:
                #df = pd.read_csv(io.TextIOWrapper(csv_file, encoding='utf-8'), delimiter=',')
                df = pd.read_csv(csv_file, delimiter='\t',header=None,names=headers)
                print(df.head())  # Display the first few rows of the DataFrame

    #df = await strat.run_simulation()
    #print(df)

if __name__ == "__main__":
    #import asyncio
    #asyncio.run(main())
    main()
