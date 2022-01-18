import jax
import jax.numpy as jnp
import numpy as np
from functools import partial



class ConfigurationSpace:
    """
    class to represent the configuration space of points in R3
    we represent a point in the configuration space as a
    number_of_points x 3 dimensional tensor.
    """
    def __init__(self, number_of_points):
        self.number_of_points = number_of_points
        self.dimension = number_of_points * 3



