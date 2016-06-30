
import theano.tensor as T
from mozi.layers.template import Template

class Scale(Template):

    """
    Scale the input into a range

    Parameters
    ----------
    X : ndarray, 2-dimensional
        numpy matrix with examples indexed on the first axis and
        features indexed on the second.

    global_max : real
        the maximum value of the whole dataset. If not provided, global_max is set to X.max()

    global_min : real
        the minimum value of the whole dataset. If not provided, global_min is set to X.min()

    scale_range : size 2 list
        set the upper bound and lower bound after scaling

    buffer : float
        the buffer on the upper lower bound such that [L+buffer, U-buffer]
    """


    def __init__(self, global_max, global_min, scale_range=[-1,1], buffer=0.1):

        self.scale_range = scale_range
        self.buffer = buffer
        self.max = global_max
        self.min = global_min
        assert scale_range[0] + buffer < scale_range[1] - buffer, \
                'the lower bound is larger than the upper bound'
        self.params = []


    def _train_fprop(self, state_below):
        width = self.max - self.min
        scale = (self.scale_range[1] - self.scale_range[0] - 2 * self.buffer) / width
        state_below = scale * (state_below - self.min)
        state_below = state_below + self.scale_range[0] + self.buffer
        return state_below
