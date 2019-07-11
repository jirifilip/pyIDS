import numpy as np

def encode_label(actual, predicted):
    levels = set(actual)
    
    actual = np.copy(actual)
    predicted = np.copy(predicted)
    
    for idx, level in enumerate(levels):
        actual[actual == level] = idx
        predicted[predicted == level] = idx

    actual = actual.astype(int)
    predicted = predicted.astype(int)
        
    return actual, predicted


def mode(array):
    values, counts = np.unique(array, return_counts=True)
    idx = np.argmax(counts)
    
    return values[idx] 