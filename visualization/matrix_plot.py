import matplotlib.pyplot as plt
import numpy as np
import os
from logger import logger

def visualize_and_save_matrix(matrix, target_language, fisher, paradigm, output_folder):
    if not os.path.isdir(output_folder):
        raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

    plt.figure(figsize=(12, 12))
    plt.imshow(matrix, cmap=plt.get_cmap('Spectral').reversed(), aspect='equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{target_language}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved correlation matrix for {target_language} to {output_path}")
    logger.info("Saved correlation matrix for %s to %s", target_language, output_path)
