from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import colt
from omegaconf import OmegaConf

from automlcli.exceptions import ConfigurationError
from automlcli.models import Model
from automlcli.settings import DEFAULT_COLT_SETTING
from automlcli.util import set_random_seed


def load_yaml(
    file_path: Union[str, Path],
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    overrides = overrides or []

    file_config = OmegaConf.load(file_path)
    overrides_config = OmegaConf.from_dotlist(overrides)

    config = OmegaConf.merge(file_config, overrides_config)
    dictconfig = OmegaConf.to_container(config)

    if not isinstance(dictconfig, dict):
        raise ConfigurationError(f"Config shoud be a dict: {file_path}")

    return dictconfig


def build_config(config: Dict[str, Any]) -> Model:
    if "automlcli" in config:
        config = config["automlcli"]
        if not isinstance(config, dict):
            raise ConfigurationError(
                "`automlcli` field is given, but it is not a dictionary."
            )

    random_seed = config.pop("random_seed", 13370)
    numpy_seed = config.pop("numpy_seed", 1337)
    set_random_seed(random_seed, numpy_seed)

    colt_config = DEFAULT_COLT_SETTING
    colt_config.update(config.pop("colt", {}))

    model_config = config["model"]
    model = colt.build(model_config, cls=Model, **colt_config)  # type: Model
    return model
