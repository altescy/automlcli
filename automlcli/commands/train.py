import argparse
import json
import logging
import pickle
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import minato
from flatten_dict import flatten

try:
    import mlflow

    # Patch mlflow.log_params to log nested params.
    _mlflow_log_param = mlflow.log_param
    _mlflow_log_params = mlflow.log_params

    def _log_params(params: Dict[str, Any]) -> None:
        flattened_params = flatten(params, reducer="dot")
        _mlflow_log_params(flattened_params)

    def _log_param(key: str, value: Any) -> None:
        if isinstance(value, dict):
            flattened_params = flatten(value, reducer="dot")
            flattened_params = {f"{key}.k": v for k, v in flattened_params.items()}
            _log_params(flattened_params)
        else:
            _mlflow_log_param(key, value)

    mlflow.log_param = _log_param
    mlflow.log_params = _log_params
except ImportError:
    mlflow = None

from automlcli.commands.subcommand import Subcommand
from automlcli.configs import build_config, load_yaml
from automlcli.util import create_workdir

logger = logging.getLogger(__name__)


@contextmanager
def _mlflow_start_run(*args: Any, **kwargs: Any) -> Iterator[Any]:
    if mlflow is not None:
        with mlflow.start_run(*args, **kwargs) as run:
            yield run
    else:
        yield None


@Subcommand.register(
    name="train",
    description="train a automl model and export the trained model",
    help="train a automl model and export the trained model",
)
class TrainCommand(Subcommand):
    def set_arguments(self) -> None:
        self.parser.add_argument(
            "config",
            type=str,
            help="path to an automl configuration file",
        )
        self.parser.add_argument(
            "train",
            type=str,
            help="path to a training data file",
        )
        self.parser.add_argument(
            "overrides",
            nargs="*",
            help="arguments to override config values",
        )
        self.parser.add_argument(
            "-s",
            "--serialization-dir",
            type=str,
            default=None,
            help="directory in which to save the model and its logs",
        )
        self.parser.add_argument(
            "--validation",
            type=str,
            default=None,
            help="path to a validation data file",
        )
        self.parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

    def run(self, args: argparse.Namespace) -> None:
        logger.info("Load config from %s", args.config)
        config = load_yaml(minato.cached_path(args.config), args.overrides)

        logger.info("Configuration: %s", str(config))
        model = build_config(config)

        logger.info("Start training...")
        logger.info("Training data: %s", args.train)
        logger.info("Validation data: %s", str(args.validation))

        with _mlflow_start_run():
            if mlflow is not None:
                logger.info("Log params to mlflow")
                params = {
                    "command": " ".join(sys.argv),
                    "config_file": args.config,
                    "train_file": args.train,
                    "validation_file": args.validation,
                    "serialization_dir": args.serialization_dir,
                    "config": config,
                }
                mlflow.log_params(params)

            serialization_dir = args.serialization_dir
            if args.serialization_dir is None and mlflow is None:
                serialization_dir = "./output"

            with create_workdir(
                serialization_dir,
                exist_ok=args.force,
            ) as workdir:
                try:
                    metrics = model.train(args.train, args.validation, workdir)
                    if mlflow is not None:
                        logger.info("Log metrics to mlflow")
                        mlflow.log_metrics(metrics)

                    logger.info("Training completed")
                    logger.info("Training metrics: %s", json.dumps(metrics, indent=2))

                    with open(workdir / "metrics.json", "w") as metrics_file:
                        json.dump(metrics, metrics_file)

                    with open(workdir / "model.pkl", "wb") as model_file:
                        pickle.dump(model, model_file)
                finally:
                    if mlflow is not None:
                        logger.info("Log metrics to mlflow")
                        mlflow.log_artifacts(str(workdir))

        logger.info("Done!")
