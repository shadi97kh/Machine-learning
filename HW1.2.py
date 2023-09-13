import numpy as np

def calculate_correlation(v1, v2):
    
    return np.corrcoef(v1, v2)[0,1]

def calculate_cosine_similarity(v1, v2):
    
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def calculate_euclidean_distance(v1, v2):
    
    return np.linalg.norm(v1-v2)

v1 = np.random.rand(10)
v2 = np.random.rand(10)

print(f"vector 1: {v1}")
print(f"vector 2: {v2}")
print(f"correlation coefficient: {calculate_correlation(v1, v2)}")
print(f"cosine similarity: {calculate_cosine_similarity(v1, v2)}")
print(f"euclidean distance: {calculate_euclidean_distance(v1, v2)}")