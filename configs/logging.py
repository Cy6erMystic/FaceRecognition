import yaml
from logging import config, getLogger

config.dictConfig(yaml.load(open("logging.yml", "r"), yaml.FullLoader))
def getLogger_(name: str = None):
    return getLogger(name)