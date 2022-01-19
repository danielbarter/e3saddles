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
    class to represent the configuration space of points in R3 we
    represent a point in the configuration space as a number_of_points
    x 3 dimensional tensor. We are particularly interested in the
    componentwise action of E3 on the configuration space. Distances
    between points and angles between segments are primitive invariant
    functions under the E3 action.
    """

    def __init__(self, number_of_points, seed=42):
        self.number_of_points = number_of_points




        # we specify a distance invariant function as a
        # tuple (i,j) where i < j < number_of_points. The
        # attribute distance_functions maps tuples to GPU
        # compiled functions implementing the distance function
        self.distance_functions_dict = {}
        self.distance_functions = []

        for i in range(number_of_points):
            for j in range(number_of_points):
                if i < j:
                    tup = (i,j)
                    func = partial(distance, tup)
                    self.distance_functions_dict[tup] = func
                    self.distance_functions.append(func)


        # we specify an angle invariant function as a
        # tuple (base,i,j) with no repeats where i < j,
        # base < number_of_points and j < number_of_points.
        # The attribute angle_functions maps tuples to GPU
        # compiled functions implementing the angle function
        self.angle_functions_dict = {}
        self.angle_functions = []

        for base in range(number_of_points):
            for i in range(number_of_points):
                for j in range(number_of_points):
                    if ( base != i and
                         base != j and
                         i < j ):

                        tup = (base,i,j)
                        func = partial(angle, tup)
                        self.angle_functions_dict[tup] = func
                        self.angle_functions.append(func)



    def random_point(self, seed):
        """
        generate a random point in the configuration space
        """

        return jax.random.normal(
            jax.random.PRNGKey(seed),
            shape=(self.number_of_points,3))


    def random_surface(self, seed, number_of_factors):
        """
        return a randomly generated surface over the configuration space
        """
        invariant_functions = self.distance_functions + self.angle_functions
        coefficients = jax.random.normal(
            jax.random.PRNGKey(seed),
            shape=(number_of_factors,len(invariant_functions)))

        def surface(point):
            result = 1
            for i in range(number_of_factors):
                accumulator = 0
                for j in range(len(invariant_functions)):
                    accumulator += coefficients[i,j] * invariant_functions[j](point)

                result *= jnp.sin(accumulator)

            return result

        return jax.jit(surface)
