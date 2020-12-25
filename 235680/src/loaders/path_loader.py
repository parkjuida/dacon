import os


def get_project_root():
    current_dir = os.path.dirname(__file__)
    return os.sep.join(current_dir.split(os.sep)[:-2])


def get_data_path():
    return f"{get_project_root()}{os.sep}data"


def get_train_data_path():
    return f"{get_data_path()}{os.sep}train"


def get_test_data_path():
    return f"{get_data_path()}{os.sep}test"
