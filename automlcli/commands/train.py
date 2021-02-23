import argparse
import logging
import pickle

from automlcli.configs import load_yaml, build_config
from automlcli.util import cached_path, open_file
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
            help="path to a trained model file",
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

    def run(self, args: argparse.Namespace) -> None:
        logger.info("Load config from %s", args.config)
        config = load_yaml(cached_path(args.config), args.overrides)

        logger.info("Configuration: %s", str(config))
        model = build_config(config)

        logger.info("Start training...")
        logger.info("Training data: %s", args.train)
        logger.info("Validation data: %s", str(args.validation))
        model.train(args.train, args.validation)

        logger.info("Training completed")

        logger.info("Save model to %s", args.output)
        with open_file(args.output, "wb") as fp:
            pickle.dump(model, fp)

        logger.info("Done!")
