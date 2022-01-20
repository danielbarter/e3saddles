import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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


@partial(jax.jit, static_argnums=[0])
def moving_frame(tup, point):

    base, i, j = tup
    base_vector = point[base]
    displacement_1_unnormalized = point[i] - point[base]
    displacement_2_unnormalized = point[j] - point[base]


    displacement_1 = (
        displacement_1_unnormalized /
        jnp.sqrt(displacement_1_unnormalized.dot(displacement_1_unnormalized)))
    displacement_2 = (
        displacement_2_unnormalized /
        jnp.sqrt(displacement_2_unnormalized.dot(displacement_2_unnormalized)))


    column_0 = displacement_1
    column_1_unnormalized = jnp.cross(displacement_1, displacement_2)
    column_1 = ( column_1_unnormalized /
                 jnp.sqrt(column_1_unnormalized.dot(column_1_unnormalized)))
    column_2 = jnp.cross(column_0, column_1)
    return jnp.vstack((
        jnp.column_stack((column_0, column_1, column_2, base_vector)),
        jnp.array([0,0,0,1])))


@jax.jit
def e3_inverse(mat):
    rotation_part = mat[0:3, 0:3]
    translation_part = mat[0:3, 3]
    transpose = jnp.transpose(rotation_part)
    return jnp.vstack(
        (jnp.concatenate(
            (transpose, - transpose.dot(translation_part.reshape((3,1)))),
            axis=1),
         jnp.array([0,0,0,1]))
    )

class ConfigurationSpace:
    """
    class to represent the configuration space of points in R3 we
    represent a point in the configuration space as a number_of_points
    x 3 dimensional tensor. We are particularly interested in the
    componentwise action of E3 on the configuration space. Distances
    between points and angles between segments are primitive invariant
    functions under the E3 action.
    """

    def __init__(self, number_of_points):
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

        return jax.random.uniform(
            jax.random.PRNGKey(seed),
            shape=(self.number_of_points,3),
            minval=-1.0,
            maxval=1.0)


    def random_surface(self, seed, layer_1_count, layer_2_count):
        """
        E3 equivariant 2 layer NN with random coefficients.
        """


        invariant_functions = self.distance_functions + self.angle_functions

        coefficients = [
            jax.random.uniform(
                jax.random.PRNGKey(seed),
                shape=(layer_1_count,len(invariant_functions)),
                minval=-1.0,
                maxval=1.0),

            jax.random.uniform(
                jax.random.PRNGKey(seed),
                shape=(layer_2_count,layer_1_count),
                minval=-1.0,
                maxval=1.0)
        ]


        def surface(point):
            inputs = jnp.array([f(point) for f in invariant_functions])
            layer_1 = jax.nn.sigmoid(coefficients[0].dot(inputs))
            layer_2 = jax.nn.sigmoid(coefficients[1].dot(layer_1))
            return layer_2.sum()


        return jax.jit(surface)




@partial(jax.jit, static_argnums=[0])
def update_minima(function, point, factor):
    """
    returns the new point, and the val / grad norm at the old point.
    """


    val = function(point)
    grad = jax.grad(function)(point)
    grad_flattened = grad.flatten()
    grad_norm = jax.numpy.sqrt(grad_flattened.dot(grad_flattened))

    new_point = point - factor * grad


    return new_point, val, grad_norm


def find_minima(
        function,
        initial_point,
        num_steps,
        factor=0.01,
        log_frequency=1000,
        minimization_report_file="/tmp/minimzation_report.pdf"
):
    """
    loop for finding minima
    """

    point = initial_point
    function_vals = np.zeros(num_steps)
    grad_norms = np.zeros(num_steps)

    for step in range(num_steps):
        point, val, grad_norm = update_minima(function, point, factor)
        function_vals[step] = val
        grad_norms[step] = grad_norm

        if step % log_frequency == 0:
            print("step:      ", step)
            print("function:  ", val)
            print("grad norm: ", grad_norm)

    fig, axs = plt.subplots(2, 1, figsize=(5,10), gridspec_kw={"height_ratios":[1,1]})
    axs[0].plot(function_vals)
    axs[0].set_title("function vals")
    axs[1].plot(grad_norms)
    axs[1].set_title("grad norms")
    fig.savefig(minimization_report_file)

    return point
