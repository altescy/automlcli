import argparse
import json
import logging
import pickle

from automlcli.configs import load_yaml, build_config
from automlcli.util import cached_path, create_workdir
from automlcli.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)


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
            "output",
            type=str,
            help="directory in which to save the model and its logs",
        )
        self.parser.add_argument(
            "overrides",
            nargs="*",
            help="arguments to override config values",
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
        config = load_yaml(cached_path(args.config), args.overrides)

        logger.info("Configuration: %s", str(config))
        model = build_config(config)

        logger.info("Start training...")
        logger.info("Training data: %s", args.train)
        logger.info("Validation data: %s", str(args.validation))

        with create_workdir(args.output, exist_ok=args.force) as workdir:
            metrics = model.train(args.train, args.validation, workdir)

            logger.info("Training completed")
            logger.info("Training metrics: %s", json.dumps(metrics, indent=2))

            with open(workdir / "metrics.json", "w") as metrics_file:
                json.dump(metrics, metrics_file)

            with open(workdir / "model.pkl", "wb") as model_file:
                pickle.dump(model, model_file)

        logger.info("Done!")
