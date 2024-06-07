import numpy as np
from sklearn.preprocessing import OneHotEncoder

alphabet = ['A', 'C', 'G', 'T']
one_hot_encoder = OneHotEncoder(categories=[alphabet], handle_unknown='error', sparse_output=True)

def get_convolution(first: np.ndarray, second: np.ndarray) -> np.array:
    second_inverted = np.flip(second, axis=0)
    closest_power_of_two = _get_closest_power_of_two_from_two_elements(len(first), len(second))
    first_transformed = _transform(first, closest_power_of_two)
    second_transformed = _transform(second_inverted, closest_power_of_two)
    hadamar_product_by_same_symbols = first_transformed * second_transformed
    sum_by_column = np.sum(hadamar_product_by_same_symbols, axis=0)
    conclusion = np.fft.ifft(sum_by_column)
    return np.real_if_close(conclusion[:len(first) + len(second) - 1], tol=1e-10)

def _get_closest_power_of_two_from_two_elements(first: int, second: int) -> int:
    return np.power(2, np.ceil(np.log2(first + second - 1)).astype(int))

def _encode(sequence: np.array, encoder: OneHotEncoder) -> np.array:
    # prepare this one for OneHotEncoder
    sequence = sequence.reshape(-1, 1)
    encoded_sequence = encoder.fit_transform(sequence).transpose().toarray()
    return encoded_sequence

def _transform(sequence: np.ndarray, closest_power_of_two: int) -> np.ndarray:
    encoded = _encode(sequence, one_hot_encoder)
    pad_width = ((0, 0), (0, closest_power_of_two - len(sequence)))
    encoded_extended = np.pad(encoded, pad_width, mode='constant')
    transformed = np.fft.fft(encoded_extended)
    return transformed
