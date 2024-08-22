"""
Main script of the project.
"""

import os
import logging
import argparse
import multiprocessing as mp
from typing import Tuple

import numpy as np

from mlp import split, train, predict
from mlp.logger import logger

SEED_DIR = './seeds'


def run_model(
    dataset_path: str,
    model_path: str,
    seed: int
) -> Tuple[int, float, float, float]:
    """
    Runs the model with the given seed.

    Args:
        dataset_path (str): The path to the dataset.
        model_path (str): The path to the model.
        seed (int): The seed to use.

    Returns:
        Tuple[int, float, float, float]: The seed, loss, accuracy, and precision of the model.
    """

    np.random.seed(seed)

    seed_dir = os.path.join(SEED_DIR, f'seed_{seed}')
    data_dir = os.path.join(seed_dir, 'data')
    model_dir = os.path.join(seed_dir, 'models')

    split(
        dataset_path=dataset_path,
        test_size=0.2,
        out_dir=data_dir
    )

    train(
        dataset_path=os.path.join(data_dir, 'train.csv'),
        model_path=model_path,
        out_dir=model_dir,
        no_plot=True,
        plot_n=0,
        plot_multi=False,
        plot_raw=False
    )

    loss, accuracy, precision = predict(
        dataset_path=os.path.join(data_dir, 'test.csv'),
        model_path=os.path.join(model_dir, 'model.json'),
    )

    return seed, loss, accuracy, precision


def set_logger_level() -> None:
    """
    Sets the logger level to ERROR.
    """

    logger.setLevel(logging.ERROR)


def update_progress(result: Tuple[int, float, float, float]) -> None:
    """
    Displays the progress of the model.

    Args:
        result (Tuple[int, float, float, float]): The result of the model.
    """

    seed, loss, accuracy, precision = result
    logger.debug(
        "seed: %4d - loss: %.4f - accuracy: %.4f - precision: %.4f",
        seed,
        loss,
        accuracy,
        precision
    )


def main() -> None:
    """
    Main function of the script.
    """

    parser = argparse.ArgumentParser(
        description='Find the best seed for the model.'
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='The path to the dataset.'
    )
    parser.add_argument(
        'model',
        type=str,
        help='The path to the model.'
    )
    args = parser.parse_args()

    seeds = range(1000)
    os.makedirs('./seed', exist_ok=True)

    logger.info('Starting the models training for %d seeds', len(seeds))
    logger.info('Number of CPUs: %d', mp.cpu_count())

    results = []
    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=set_logger_level
    ) as pool:
        for seed in seeds:
            result = pool.apply_async(
                run_model,
                args=(args.dataset, args.model, seed,),
                callback=update_progress
            )
            results.append(result)

        pool.close()
        pool.join()

    results = [r.get() for r in results]
    best_seed, best_loss, best_accuracy, best_precision = min(results, key=lambda x: x[1])

    logger.info('All models have been trained.')
    logger.info(
        'Best Seed: %d with Loss: %.4f - Accuracy: %.4f - Precision: %.4f',
        best_seed,
        best_loss,
        best_accuracy,
        best_precision
    )


if __name__ == '__main__':
    main()
