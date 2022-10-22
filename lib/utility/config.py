import yaml


class Config:
    def __init__(self, path):
        with open(path) as file:
            self.config = yaml.safe_load(file.read())

    def __getattr__(self, item):
        return self.config.get(item, None)
