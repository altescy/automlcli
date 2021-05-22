import argparse
import logging
import pickle

import minato

from automlcli.commands.subcommand import Subcommand
from automlcli.models import Model

logger = logging.getLogger(__name__)


@Subcommand.register(
    name="retrain",
    description="retrain model with new data",
    help="retrain model with new data",
)
class RetrainCommand(Subcommand):
    def set_arguments(self) -> None:
        self.parser.add_argument(
            "model",
            type=str,
            help="path to an automl configuration file",
        )
        self.parser.add_argument(
            "data",
            type=str,
            help="path to a training data file",
        )
        self.parser.add_argument(
            "output",
            type=str,
            default=None,
            help="path to a output file of retrained model",
        )

    def run(self, args: argparse.Namespace) -> None:
        logger.info("Load model from %s", args.model)
        with minato.open(args.model, "rb") as fp:
            model = pickle.load(fp)  # type: Model

        logger.info("Retrain model with %s", args.data)
        model.retrain(args.data)

        with minato.open(args.output, "wb") as fp:
            pickle.dump(model, fp)

        logger.info("Done!")
