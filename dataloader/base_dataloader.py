import os
from utils import printf

def mkdirp(logdir):
    if not os.path.exists(logdir):
        os.mkdir(logdir)

class DataLoader(object):
    ############################################################################
    # Initialization
    def __init__(self):
        self.printer = self.initialize_printer(None, None)

    def initialize_printer(self, logger, args):
        if logger is None:
            assert args is None
            def printer(x):
                print x
        else:
            printer = lambda x: printf(logger, args, x)
        self.printer = printer

    def initialize_data(self):
        raise NotImplementedError

    def create_dataset_name(self):
        raise NotImplementedError

    ############################################################################
    # Data Generation
    def generate_unique_dataset(self):
        pass

    def insufficient_coverage(self):
        pass

    def save_dataset(self):
        pass

    def save_datasets(self):
        pass

    ############################################################################
    # Loading data
    def load_dataset(self):
        pass

    def preprocess(self, problem):
        raise NotImplementedError

    def record_state(self, state):
        raise NotImplementedError

    def load_problem(self, mode):
        raise NotImplementedError

    def reset(self, mode):
        raise NotImplementedError

    ############################################################################
    # Curriculum
    def add_dataset(self, mode):
        raise NotImplementedError

    def update_curriculum(self):
        raise NotImplementedError

    ############################################################################
    # Visualization
    def get_trace(self):
        raise NotImplementedError
