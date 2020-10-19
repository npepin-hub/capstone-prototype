
from dynaconf import Dynaconf
import logging
import logging.config
import yaml
import os

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load this files in the order.
    
def setupLogging():
    dirname = os.path.dirname(__file__)
    log_file_path = dirname+"/"+settings.log_config 
    with open(log_file_path, 'rt') as file:
        config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)


setupLogging()