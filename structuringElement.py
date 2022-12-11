import numpy as np
class StructuringElement:
    def __init__(self, kernel: np.ndarray, seed: tuple) -> None:
        self.kernel = kernel
        self.seed = seed

    def number_of_matches(self, mat: np.ndarray, i: int, j: int, max_level: int, padding: int):
        n, m = mat.shape
        nn, mm = self.kernel.shape
        kernel = self.kernel * max_level
        ret = 0
        for r in range(nn):
            for c in range(mm):
                row = i + r
                col = j + c
                if row < 0 or row >= n or col < 0 or col >= m:
                    val = padding * max_level
                else:
                    val = mat[row][col]
                if val == kernel[r][c]:
                    ret += 1

        return ret
    
    def matches(self, mat: np.ndarray, i: int, j: int, max_level: int, padding: int):
        matches_number = self.number_of_matches(mat, i, j, max_level, padding)
        nn, mm = self.kernel.shape
        return matches_number == nn * mm
    
    def has_match(self, mat: np.ndarray, i: int, j: int, max_level: int, padding: int):
        return self.number_of_matches(mat, i, j, max_level, padding) > 0
