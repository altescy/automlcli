import argparse
import json
import logging
import pickle

from sklearn.metrics import get_scorer

from automlcli.models import Model
from automlcli.util import open_file
from automlcli.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)


@Subcommand.register(
    name="evaluate",
    description="evaluate trained model",
    help="evaluate trained model",
)
class EvaluateCommand(Subcommand):
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
            "--scoring",
            type=str,
            action="append",
            default=[],
            help="evaluation metrics",
        )
        self.parser.add_argument(
            "--cv",
            type=int,
            default=None,
            help="cross validation strategy",
        )
        self.parser.add_argument(
            "--output-file",
            type=str,
            default=None,
            help="path to a output file of evaluation result",
        )
        self.parser.add_argument(
            "--quiet",
            action="store_true",
            help="do not show evaluation result",
        )

    def run(self, args: argparse.Namespace) -> None:
        logger.info("Load model from %s", args.model)
        with open_file(args.model, "rb") as fp:
            model = pickle.load(fp)  # type: Model

        logger.info("Model: %s", model)

        logger.info("Load data from %s", args.data)
        X, y = model.load_data(args.data)
        if y is None:
            raise ValueError(f"Target column not found: {args.data}")

        logger.info("Evaluate model")
        scoring = args.scoring or ["accuracy"]
        scorers = {scoring: get_scorer(scoring) for scoring in scoring}
        if args.cv is None:
            metrics = {
                metric: scorer(model, X, y)
                for metric, scorer in scorers.items()
            }
        else:
            raise NotImplementedError("Cross validation is not implemented.")

        if not args.quiet:
            print(json.dumps(metrics, indent=2))

        if args.output_file is not None:
            logger.info("Save predictions to %s", args.output_file)
            with open_file(args.output_file, "w") as fp:
                json.dump(metrics, fp)

        logger.info("Done!")
