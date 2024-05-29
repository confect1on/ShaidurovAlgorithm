import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ShaidurovAlgorithm:
    def __init__(self, alphabet=None):
        if alphabet is None:
            alphabet = ['A', 'C', 'G', 'T']
        self.alphabet = alphabet
        self.one_hot_encoder = OneHotEncoder(categories=[self.alphabet], handle_unknown='error', sparse_output=True)

    def get_conclusion(self, first: np.ndarray, second: np.ndarray) -> np.array:
        second_inverted = np.flip(second, axis=0)
        closest_power_of_two = self._get_closest_power_of_two_from_two_elements(len(first), len(second))
        first_transformed = self._transform(first, closest_power_of_two)
        second_transformed = self._transform(second_inverted, closest_power_of_two)
        hadamar_product_by_same_symbols = first_transformed * second_transformed
        sum_by_column = np.sum(hadamar_product_by_same_symbols, axis=0)
        conclusion = np.fft.ifft(sum_by_column)
        return conclusion[:len(first) + len(second) - 1]

    @staticmethod
    def _get_closest_power_of_two_from_two_elements(first: int, second: int) -> int:
        return np.power(2, np.ceil(np.log2(first + second - 1)).astype(int))

    @staticmethod
    def _encode(sequence: np.array, encoder: OneHotEncoder) -> np.array:
        # prepare this one for OneHotEncoder
        sequence = sequence.reshape(-1, 1)
        encoded_sequence = encoder.fit_transform(sequence).transpose().toarray()
        return encoded_sequence

    def _transform(self, sequence: np.ndarray, closest_power_of_two: int) -> np.ndarray:
        encoded = self._encode(sequence, self.one_hot_encoder)
        pad_width = ((0, 0), (0, closest_power_of_two - len(sequence)))
        encoded_extended = np.pad(encoded, pad_width, mode='constant')
        transformed = np.fft.fft(encoded_extended)
        return transformed
