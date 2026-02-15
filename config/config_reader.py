import yaml
import os
from src.utils.general_utils import print_x
from yaml import Loader


def read_config(file_path=None):
    if not file_path:
        current_working_dir = os.getcwd()
        file_path = os.path.join(current_working_dir, "config/base.yaml")
    print_x(f"{os.path.abspath(file_path)}")
    with open(file_path, "r") as f:
        cfg = yaml.load(f, Loader=Loader)
        f.close()
        return cfg
