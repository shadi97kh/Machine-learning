import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine, euclidean

def calculate_correlation(v1, v2):

    return pearsonr(v1, v2)[0]

def calculate_cosine_similarity(v1, v2):

    return 1 - cosine(v1, v2)

def calculate_euclidean_distance(v1, v2):

    return euclidean(v1, v2)

v1 = np.random.rand(10)
v2 = np.random.rand(10)

v1_series = pd.Series(v1)
v2_series = pd.Series(v2)
correlation_pandas = v1_series.corr(v2_series)

print(f"vector 1: {v1}")
print(f"vector 2: {v2}")
print(f"Correlation Coefficient (SciPy): {calculate_correlation(v1, v2)}")
print(f"Correlation Coefficient (pandas): {correlation_pandas}")
print(f"Cosine Similarity: {calculate_cosine_similarity(v1, v2)}")
print(f"Euclidean Distance: {calculate_euclidean_distance(v1, v2)}")
