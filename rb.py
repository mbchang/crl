from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
Transition = namedtuple('Transition', ('state', 'action', 'logprob', 'mask', 'next_state', 'reward', 'value'))
SimpleTransition = namedtuple('SimpleTransition', ('state', 'action', 'logprob', 'mask','reward', 'value'))
InputOutput = namedtuple('InputOutput', ('loss'))
RNTransition = namedtuple('StepTransition', ('state', 'action', 'logprob', 'mask', 'reward', 'value', 'step', 'task'))

class Memory(object):
    def __init__(self, element='transition'):
        self.memory = []
        if element == 'transition':
            self.element = Transition
        elif element == 'simpletransition':
            self.element = SimpleTransition
        elif element == 'inputoutput':
            self.element = InputOutput
        elif element == 'rntransition':
            self.element = RNTransition
        else:
            assert False

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.element(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.element(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.element(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def clear_buffer(self):
        del self.memory[:]
