import numpy as np

def fisher_transform(correlation_matrix):
    clipped_matrix = np.clip(correlation_matrix, -0.9999, 0.9999)
    return np.arctanh(clipped_matrix)

def calculate_region_averages(matrix):
    mask = np.eye(matrix.shape[0], dtype=bool)
    Language_region = matrix[:12, :12]
    MD_region = matrix[12:, 12:]
    Lang_MD_top_right = matrix[:12, 12:]
    Lang_MD_bottom_left = matrix[12:, :12]
    Lang_MD_region = np.concatenate((Lang_MD_top_right.flatten(), Lang_MD_bottom_left.flatten()))

    Language_mask = ~mask[:12, :12]
    MD_mask = ~mask[12:, 12:]

    return np.mean(Language_region[Language_mask]), np.mean(MD_region[MD_mask]), np.mean(Lang_MD_region)
