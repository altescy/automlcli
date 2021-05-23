import argparse
import logging
import pickle
import sys

import minato

from automlcli.commands.subcommand import Subcommand
from automlcli.models import Model

logger = logging.getLogger(__name__)


@Subcommand.register(
    name="predict",
    description="make predition by the trained model",
    help="make predition by the trained model",
)
class PredictCommand(Subcommand):
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
            "--output-file",
            type=str,
            default=None,
            help="path to a output file of prediction",
        )
        self.parser.add_argument(
            "--output-column",
            type=str,
            default=None,
            help="column name of prediction",
        )
        self.parser.add_argument(
            "--quiet",
            action="store_true",
            help="do not show predicted results",
        )

    def run(self, args: argparse.Namespace) -> None:
        logger.info("Load model from %s", args.model)
        with minato.open(args.model, "rb") as fp:
            model = pickle.load(fp)  # type: Model

        logger.info("Make predictions for %s", args.data)
        predictions = model.predict(
            args.data,
            prediction_column=args.output_column,
        )

        index = model.index_column is not None

        if not args.quiet:
            predictions.to_csv(sys.stdout, index=index)

        if args.output_file is not None:
            logger.info("Save predictions to %s", args.output_file)
            with minato.open(args.output_file, "wb") as fp:
                predictions.to_csv(fp, index=index)

        logger.info("Done!")
