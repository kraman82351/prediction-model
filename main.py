# Import the function from data_processing.py
from kfoldData import read_csv_and_kfold_segment
from prediction_model import training_testing
import random

random_number = random.randint(0, 5)

# Call the function with the desired CSV file and segment number
segment_data = read_csv_and_kfold_segment(random_number)
print("segmented Data Retrieved")
training_testing(segment_data)
