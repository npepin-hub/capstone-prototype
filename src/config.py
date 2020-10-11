
from dynaconf import Dynaconf
import logging
import logging.config
import yaml

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load this files in the order.


def setupLogging():
    log_file_path = settings.log_config 
    with open(log_file_path, 'rt') as file:
        config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)


setupLogging()