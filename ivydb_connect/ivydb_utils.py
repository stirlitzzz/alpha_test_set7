import json
from typing import Dict, List, Any
import pandas as pd

class IvyDBUtils:
    def __init__(self, definitions_file: str):
        with open(definitions_file, 'r') as f:
            self.table_definitions = json.load(f)

    def get_table_names(self) -> List[str]:
        """Return a list of all table names."""
        return list(self.table_definitions.keys())

    def get_table_structure(self, table_name: str) -> List[Dict[str, str]]:
        """Return the structure of a specific table."""
        return self.table_definitions.get(table_name, [])

    def get_column_names(self, table_name: str) -> List[str]:
        """Return a list of column names for a specific table."""
        return [col['name'] for col in self.table_definitions.get(table_name, [])]

    def get_dtype_dict(self, table_name: str) -> Dict[str, Any]:
        """Return a dictionary of column names and their corresponding pandas dtype."""
        dtype_map = {
            'integer': 'int64',
            'bigint': 'int64',
            'real': 'float64',
            'decimal': 'float64',
            'char': 'object',
            'date': 'datetime64[ns]'
        }
        
        return {
            col['name']: dtype_map.get(col['type'].split('(')[0], 'object')
            for col in self.table_definitions.get(table_name, [])
            if col['type'] != 'date'  # Exclude date columns as they'll be parsed separately
        }

    def read_file(self, file_path: str, table_name: str) -> pd.DataFrame:
        """Read an IvyDB file into a pandas DataFrame using the appropriate table definition."""
        dtype_dict = self.get_dtype_dict(table_name)
        date_columns = [col['name'] for col in self.table_definitions.get(table_name, []) if col['type'] == 'date']
        
        df = pd.read_csv(
            file_path,
            delimiter='\t',
            names=self.get_column_names(table_name),
            dtype=dtype_dict,
            parse_dates=date_columns
        )
        
        # Handle missing values (assuming -99.99 is used for missing values)
        df = df.replace(-99.99, pd.np.nan)
        
        return df

# Example usage
if __name__ == "__main__":
    ivy_db = IvyDBUtils('ivydb_table_definitions.json')
    
    print("Available tables:", ivy_db.get_table_names())
    
    print("\nStructure of Security_Price table:")
    for column in ivy_db.get_table_structure('Security_Price'):
        print(f"{column['name']} ({column['type']}): {column['description']}")
    
    print("\nReading a Security_Price file:")
    df = ivy_db.read_file('path_to_your_file.txt', 'Security_Price')
    print(df.head())
    print(df.dtypes)
