from butadien.load_data import load_data
from SCFInitialGuess.utilities.dataset import Dataset

dim = 26
source = "notebooks/butadien/data"
dataset = Dataset(*load_data(source, 10)) 

print(dataset)