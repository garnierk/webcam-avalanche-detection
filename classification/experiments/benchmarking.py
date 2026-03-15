import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import comet_ml

import pandas as pd
from .experiment_run import ExperimentRun
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, VGG13_Weights,
                                VGG16_Weights, VGG19_Weights)

IM_SIZE = 704
'''Network input size'''

DEFAULT_DATA_ROOT = '.data'
'''Default folder containing /train and /test image folders'''

DEFAULT_PROJECT_DIR = './classification/experiments'
'''Default project output directory (contains model folders and benchmark csv)'''


def _build_paths(project_dir: str) -> Tuple[str, str]:
    '''Build output paths from the project output directory.'''
    base_path = Path(project_dir)
    models_dir = str(base_path / 'models' / f'benchmarks_{IM_SIZE}')
    results_file = str(base_path / f'benchmark_scores_{IM_SIZE}.csv')
    return models_dir, results_file


def _base_cfg(data_root: str) -> Dict[str, Any]:
    '''Default config to use for training runs.'''
    data_root = data_root.rstrip('/\\')
    return {
        'batch_size': 10,
        'epochs': 40,
        'image_loader': None,
        'lr': 2.25e-5,
        'num_classes': 4,
        'num_workers': 8,
        'optimizer': 'Adam',
        'test_dir': f'{data_root}/test/images',
        'train_dir': f'{data_root}/train/images',
        'train_transforms': None,
        'use_comet': True,
        'input_size': IM_SIZE,
        'full_size': int(IM_SIZE * 1.05),
    }

BENCHMARK_MODELS: List[Tuple[str, Any]] = [
    ('ResNet152', ResNet152_Weights.IMAGENET1K_V2),
    #('ResNet101', ResNet101_Weights.IMAGENET1K_V2),
    #('ResNet50', ResNet50_Weights.IMAGENET1K_V2),
    #('ResNet34', ResNet34_Weights.IMAGENET1K_V1),
    #('ResNet18', ResNet18_Weights.IMAGENET1K_V1),
    #('vgg19', VGG19_Weights.IMAGENET1K_V1),
    #('vgg16', VGG16_Weights.IMAGENET1K_V1),
    #('vgg13', VGG13_Weights.IMAGENET1K_V1),
]


def _model_name(architecture: str) -> str:
    '''Model name for a given architecture'''
    return f'{architecture}_benchmark_{IM_SIZE}'


def train_model(architecture: str, weights, base_cfg: Dict[str, Any], models_dir: str,
                comet_project: str = 'avalanche_benchmark'):
    '''Train a model for the given architecture and weights.

    Args:
        architecture (str):  the model base architecture.
        weights:             PyTorch weights for the given base architecture.
        comet_project (str): name of the comet project for logging results.
    '''
    os.makedirs(models_dir, exist_ok=True)
    model_name = _model_name(architecture)
    train_config = {
        **base_cfg,
        'architecture': architecture,
        'weights': weights,
        'comet_init': {'project': comet_project,
                       'name': model_name,
                       'tags': [architecture, 'Adam', str(weights)]}
    }
    Experiment = ExperimentRun(run_dir=models_dir, experiment_name=model_name)
    Experiment.start_training(
        n_runs=3, train_config=train_config, description=f'Benchmark experiment for {architecture} with weights {weights}')


def eval_benchmark_models(models_dir: str, results_file: str,
                          score_keys=['test/accuracy', 'test/f1', 'test/binary/accuracy',
                                      'test/binary/f1', 'test/weighted_accuracy', 'test/weighted_f1',
                                      'test/binary/weighted_accuracy', 'test/binary/weighted_f1']):
    '''Evaluate trained models and save scores to a .csv file

    Args:
        score_keys (list[str]): list of score keys to save values for'''
    model_key = 'model'

    scores: Dict[str, List[Union[float, str]]] = {
        model_key: []} | {f'{key}/mean': [] for key in score_keys} | {f'{key}/std': [] for key in score_keys}

    # Calculate scores for each model
    for (architecture, _) in BENCHMARK_MODELS:
        model_name = _model_name(architecture)
        _, score_summary = _get_score_from_model_name(
            model_name, models_dir=models_dir, score_keys=score_keys)
        print(score_summary)

        scores[model_key].append(architecture)
        for key in score_keys:
            scores[f'{key}/mean'].append(f'{score_summary[key]["mean"]:.1f}')
            scores[f'{key}/std'].append(f'{score_summary[key]["std"]:.1f}')

    # Save scores as a csv
    results_dir = os.path.dirname(results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    output_df = pd.DataFrame(scores)
    output_df.to_csv(results_file, index=False)


def _get_score_from_model_name(model_name: str, suppress_logging: bool = True,
                               models_dir: str = '', test_dir=None, train_dir=None,
                               score_keys: List[str] = []) -> Dict[str, float]:
    '''Load a model and return the mean scores with keys in score_keys

    Args:
        model_name (str): the model to load.
        suppress_logging (bool): if false then print verbose output.
        test_dir (str): optionally override the test directory saved in the model config.
        train_dir (str): optionally override the train directory saved in the model config.
        score_keys (list[str]): the metrics to save for the model.
    '''
    Experiment = ExperimentRun(
        run_dir=models_dir, experiment_name=model_name)
    _, score_summary = Experiment.evaluate(score_keys=score_keys, early_stopping=True,
                                           suppress_logging=suppress_logging, test_dir=test_dir, train_dir=train_dir)
    print(score_summary)
    return ({key: f"{score_summary[key]['mean']:.2f}" for key in score_keys}), score_summary


def _parse_args():
    parser = argparse.ArgumentParser(description='Run classification benchmark experiments.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_DATA_ROOT,
        help='Path to dataset root containing train/images and test/images.',
    )
    parser.add_argument(
        '--project-dir',
        type=str,
        default=DEFAULT_PROJECT_DIR,
        help='Output project directory for benchmark models and score CSV.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    models_dir, results_file = _build_paths(args.project_dir)
    os.makedirs(models_dir, exist_ok=True)
    base_cfg = _base_cfg(args.data_dir)

    # Train models
    for (architecture, weights) in BENCHMARK_MODELS:
        train_model(architecture, weights, base_cfg=base_cfg, models_dir=models_dir)

    # Evaluate models
    eval_benchmark_models(models_dir=models_dir, results_file=results_file)
