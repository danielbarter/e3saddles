import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

        key = jax.random.PRNGKey(seed)

        return jax.random.uniform(
            key,
            shape=(self.number_of_points,3),
            minval=-1.0,
            maxval=1.0)


    def random_surface(self, seed, layer_1_count, layer_2_count, scale):
        """
        E3 equivariant 2 layer NN with random coefficients.
        """


        invariant_functions = self.distance_functions + self.angle_functions

        key_0 = jax.random.PRNGKey(seed)
        key_1 = jax.random.split(key_0)[1]
        key_2 = jax.random.split(key_1)[1]

        coefficients = [
            jax.random.uniform(
                key_0,
                shape=(layer_1_count,len(invariant_functions)),
                minval=-1.0,
                maxval=1.0),

            jax.random.uniform(
                key_1,
                shape=(layer_2_count,layer_1_count),
                minval=-1.0,
                maxval=1.0),

            jax.random.uniform(
                key_2,
                shape=(layer_2_count,),
                minval=-1.0,
                maxval=1.0)


        ]


        def surface(point):
            inputs = jnp.array([f(point) for f in invariant_functions])
            layer_1 = jax.nn.sigmoid(coefficients[0].dot(inputs))
            layer_2 = jax.nn.sigmoid(coefficients[1].dot(layer_1))
            return scale * coefficients[2].dot(layer_2)


        return jax.jit(surface)


### finding minima ###

@partial(jax.jit, static_argnums=[0])
def update_minima(function, point, factor):
    """
    returns the new point, and the val / grad norm at the old point.
    """


    val = function(point)
    grad = jax.grad(function)(point)
    grad_norm = jax.numpy.sqrt((grad * grad).sum())

    new_point = point - factor * grad


    return new_point, val, grad_norm


def find_minima(
        function,
        initial_point,
        num_steps,
        factor,
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


### finding geodesics ###

@partial(jax.jit, static_argnums=[0])
def action(function, left_point, right_point, distance_factor):

    displacement = right_point - left_point
    squares = displacement * displacement
    graph_component = (function(right_point) - function(left_point)) ** 2
    return jnp.exp(distance_factor * squares.sum()) - 1.0 +  graph_component


@partial(jax.jit, static_argnums=[0])
def lagrangian(
        function,      # function defining graph
        points,        # n points
        start,         # start point. fixed
        end,           # end point. fixed
        distance_factor
):

    accumulator = action(function, start, points[0], distance_factor)

    accumulator += sum(jnp.array(
        [action(function, points[i], points[i+1], distance_factor)
         for i in range(0, points.shape[0] - 1)]))

    accumulator += action(function, points[-1], end, distance_factor)

    return accumulator


@partial(jax.jit, static_argnums=[0])
def update_geodesic(function, points, start, end, factor, distance_factor):
    val = lagrangian(function, points, start, end, distance_factor)
    new_points = points -  factor * jax.grad(lagrangian, argnums=1)(function, points, start, end, distance_factor)
    return new_points, val



def compute_initial_points(start, end, number_of_points):
    ts = np.linspace(0.0, 1.0, number_of_points+1)[1:]
    points = [ start * ( 1 - t ) + end * t for t in ts ]
    return jnp.stack(points)

def find_geodesic(
        function,
        initial_points,
        start,
        end,
        num_steps,
        factor,
        distance_factor,
        log_frequency=1000,
        geodesic_report_file="/tmp/geodesic_report.pdf"
):
    result = []
    points = initial_points
    result.append(points)
    lagrangian_vals = np.zeros(num_steps)


    for step in range(num_steps):
        points, val = update_geodesic(function, points, start, end, factor, distance_factor)
        lagrangian_vals[step] = val
        if step % 1000 == 0:
            result.append(points)
            print("step:      ", step)
            print("lagrangian:", val)


    result.append(points)
    function_vals = np.zeros(points.shape[0])
    grad_norms = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        grad = jax.grad(function)(points[i]).flatten()
        grad_norms[i] = jnp.sqrt(grad.dot(grad))
        function_vals[i] = function(points[i])

    fig, axs = plt.subplots(3, 1, figsize=(5,15), gridspec_kw={"height_ratios":[1,1,1]})
    axs[0].plot(lagrangian_vals)
    axs[0].set_title("lagrangian vals")

    axs[1].plot(grad_norms)
    axs[1].set_title("geodesic grads")

    axs[2].plot(function_vals)
    axs[2].set_title("function vals above geodesic")

    fig.savefig(geodesic_report_file)

    return result




@partial(jax.jit, static_argnums=[0,1,2,3,4])
def contour_vals(function,x_min, x_max, y_min, y_max):
    x_vals = jnp.arange(x_min, x_max, 0.01)
    y_vals = jnp.arange(y_min, y_max, 0.01)
    l,r = jnp.meshgrid(x_vals, y_vals)
    args = jnp.stack([l,r],axis=2)
    return x_vals, y_vals, jnp.apply_along_axis(function, 2, args)

def contour_2d(
        function,
        x_min,
        x_max,
        y_min,
        y_max,
        levels,
        paths,
        title,
        contour_file="/tmp/contour_file.gif"):

    x_vals, y_vals, z_vals = contour_vals(function, x_min, x_max, y_min, y_max)

    ims = []
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.contour(x_vals, y_vals, z_vals, levels=levels)

    for path in paths:

        scatter = ax.scatter(path[:,0], path[:,1],color=['red'])
        ims.append([scatter])

    ani = animation.ArtistAnimation(fig, ims)
    ani.save(contour_file)

