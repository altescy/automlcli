from pathlib import Path

# colt settings
DEFAULT_COLT_SETTING = {
    "typekey": "type",
}

# automlcli directory settings
AUTOMLCLI_ROOT = Path.home() / ".automlcli"

# plugin settings
LOCAL_PLUGINS_FILENAME = ".automlcli_plugins"
GLOBAL_PLUGINS_FILENAME = AUTOMLCLI_ROOT / "plugins"
