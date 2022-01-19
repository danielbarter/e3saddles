import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=[0])
def distance(tup, point):

    i, j = tup
    displacement = point[j] - point[i]
    return displacement.dot(displacement)



@partial(jax.jit, static_argnums=[0])
def angle(tup, point):

    base, i, j = tup
    displacement_1 = point[i] - point[base]
    displacement_2 = point[j] - point[base]
    return displacement_1.dot(displacement_2)


class ConfigurationSpace:
    """
    class to represent the configuration space of points in R3
    we represent a point in the configuration space as a
    number_of_points x 3 dimensional tensor.
    """

    def __init__(self, number_of_points, seed=42):
        self.number_of_points = number_of_points
        self.dimension = number_of_points * 3
        self.random_key = jax.random.PRNGKey(seed)


        # we specify a distance invariant function as a
        # tuple (i,j) where i < j < number_of_points
        self.distance_functions = {}

        for i in range(number_of_points):
            for j in range(number_of_points):
                if i < j:
                    tup = (i,j)
                    self.distance_functions[tup] = partial(distance, tup)


        # we specify an angle invariant function as a
        # tuple (base,i,j) with no repeats where i < j,
        # base < number_of_points and j < number_of_points
        self.angle_functions = {}

        for base in range(number_of_points):
            for i in range(number_of_points):
                for j in range(number_of_points):
                    if ( base != i and
                         base != j and
                         i < j ):

                        tup = (base,i,j)
                        self.angle_functions[tup] = partial(angle, tup)



    def random_point(self):
        """
        generate a random point in the configuration space
        """

        result = jax.random.normal(
            self.random_key,
            shape=(self.number_of_points,3))
        self.random_key = jax.random.split(self.random_key)[1]
        return result

