import argparse
import json
import logging
import pickle
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import minato
import yaml
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
from automlcli.configs import ConfigBuilder, load_yaml
from automlcli.exceptions import ConfigurationError
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
            nargs="?",
            help="path to a training data file",
        )
        self.parser.add_argument(
            "--overrides",
            action="append",
            default=[],
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
        builder = ConfigBuilder.build(config)
        model = builder.model
        train_file = args.train or builder.train_file
        validation_file = args.validation or builder.validation_file

        if not train_file:
            raise ConfigurationError("train file is required.")

        logger.info("Start training...")
        logger.info("Training data: %s", str(train_file))
        logger.info("Validation data: %s", str(validation_file))

        params = {
            "command": " ".join(sys.argv),
            "config_file": args.config,
            "train_file": train_file,
            "validation_file": validation_file,
            "serialization_dir": args.serialization_dir,
            "config": config,
        }

        with _mlflow_start_run():
            serialization_dir = args.serialization_dir
            if args.serialization_dir is None and mlflow is None:
                serialization_dir = "./output"

            with create_workdir(
                serialization_dir,
                exist_ok=args.force,
            ) as workdir:
                workdir = workdir.absolute()
                try:
                    with open(workdir / "config.yaml", "w") as f:
                        yaml.dump(config, f)

                    with open(workdir / "params.json", "w") as f:
                        json.dump(params, f, indent=2)

                    if mlflow is not None:
                        logger.info("Log params to mlflow")
                        mlflow.log_params(params)

                    metrics = model.train(train_file, validation_file, workdir)

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
