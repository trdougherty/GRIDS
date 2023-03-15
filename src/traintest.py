import os
import sys
import numpy as np

def split(footprint_data, test_percent:int = 15, seed: int = 0):
    """Splits the buildings into train and test"""
    np.random.seed(seed)
    all_buildings = footprint_data.id.unique()
    test_buildings = np.random.choice(
        all_buildings, 
        int(len(all_buildings) // (1 / (test_percent/100))),
        replace=False
    )

    # validate_buildings = np.random.choice(
    #     list(filter(lambda x: x not in test_buildings, all_buildings)), 
    #     len(all_buildings) // 10,
    #     replace=False
    # )

    train_buildings = list(filter(
        lambda x: x not in [ *test_buildings ], 
        all_buildings
    ))

    return {
        "train": train_buildings,
        "test": test_buildings
    }