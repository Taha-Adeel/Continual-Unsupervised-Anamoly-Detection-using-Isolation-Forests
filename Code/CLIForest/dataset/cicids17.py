import pandas as pd
import numpy as np
import os

class CICIDS17:
    def __init__(self, path = '../datasets/CIC-IDS-17/', verbose = False):
        self.path = path
        self.filenames = os.listdir(path)
        self.datasets = []
        for filename in self.filenames:
            if verbose: 
                print(f'\nLoading {filename}...')
            data = pd.read_csv(os.path.join(path, filename))
            data = self.clean_data(data, verbose)
            X, y = self.preprocess_data(data, verbose)
            contamination = y.value_counts()[1] / y.count()
            self.datasets.append((X, y, contamination, filename))
            if verbose: 
                print(f'Loaded {filename}: \n\tShape: ({X.shape[0]}, {X.shape[1]}) \n\tNumber of anomalies: {y.value_counts()[1]} \n\tAnamoly rate: {contamination*100:.4f}%')

        self.datasets = sorted(self.datasets, key=lambda x: x[2])
        
        # Combine all datasets
        X_combined = pd.concat([dataset[0] for dataset in self.datasets])
        y_combined = pd.concat([dataset[1] for dataset in self.datasets])
        self.datasets.append((X_combined, y_combined, y_combined.value_counts()[1] / y_combined.count(), 'combined'))
        if verbose: 
            print(f'\nCombined datasets: \n\tShape: ({X_combined.shape[0]}, {X_combined.shape[1]}) \n\tNumber of anomalies: {y_combined.value_counts()[1]} \n\tAnamoly rate: {y_combined.value_counts()[1] / y_combined.count()*100:.4f}%')
            
            
    def clean_data(self, data, verbose=False):
        if verbose: 
            print(f'Cleaning: {data.shape[0]} rows and {data.shape[1]} columns...')
        data.columns = data.columns.str.strip()
        data = data.replace('Infinity', pd.NA)
        data = data.replace(float('inf'), pd.NA)
        data = data.dropna()
        for column in data.columns:
            if column != 'Label':
                data[column] = pd.to_numeric(data[column], errors='coerce', downcast='float')
                
        if verbose: 
            print(f'Cleaned: {data.shape[0]} rows and {data.shape[1]} columns.')
        
        return data
        
    def preprocess_data(self, data, verbose=False):
        if verbose: 
            print(f'Preprocessing {data.shape[0]} rows and {data.shape[1]} columns...')
        # data.drop(['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp', 'Attempted Category'], axis=1, inplace=True)
        data.drop(['Destination Port'], axis=1, inplace=True)
        if verbose: 
            print(f'Columns dropped: Destination Port')
        
        # Split data into X and y
        X = data.drop(['Label'], axis=1)
        y = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        return X, y
        
        # Drop columns with pearson correlation > 0.9
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        X = X.drop(to_drop, axis=1)
        if verbose: 
            print(f"Correlated columns dropped: {to_drop}")
            print(f'Preprocessed {X.shape[0]} rows and {X.shape[1]} columns.')
        
        return X, y
    
def test():
    dataset = CICIDS17(path='../../datasets/CIC-IDS-17/', verbose=True)
    
if __name__ == '__main__':
    test()