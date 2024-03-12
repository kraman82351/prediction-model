import pandas as pd
from sklearn.model_selection import KFold

def read_csv_and_kfold_segment(segment):
    # Read the CSV file
    data = pd.read_csv('C:/Users/Aman/Desktop/claim management system/data_science/Health Claims/merged.csv')
    
    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    segment_data = None  # Initialize segment_data variable
    
    # Get the indices for the specified segment
    for i, (_, test_index) in enumerate(kf.split(data)):
        if i == segment:
            segment_data = data.iloc[test_index]
            break
    
    return segment_data
