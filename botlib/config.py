import yaml

# Settings dict — populated by domain package during init
settings = {}


def load_settings(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
