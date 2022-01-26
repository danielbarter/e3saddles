from e3saddles.configuration_space import *

@jax.jit
def wolfe_schlegel(point):
    x = point[0]
    y = point[1]
    return 10 * (x**4 + y**4 - 2 * x*x - 4 * y*y + x * y + 0.2 * x + 0.1 * y)



minima_1 = find_minima(wolfe_schlegel, jnp.array([-1.5, 1.5]), 50000, 0.0001)
minima_2 = find_minima(wolfe_schlegel, jnp.array([-1, -1.5]),  50000, 0.0001)
minima_3 = find_minima(wolfe_schlegel, jnp.array([1.0, -1.5]),  50000, 0.0001)
special_point = jnp.array([1.0, 1.0])

initial_points_1 = compute_initial_points(minima_1, special_point, 25)
initial_points_2 = compute_initial_points(special_point, minima_3, 25)
initial_points = jnp.vstack([initial_points_1, initial_points_2])


geodesic = find_geodesic(wolfe_schlegel, initial_points, minima_1, minima_3, 50000, 0.00001, 100)
contour_2d(wolfe_schlegel, -2.0, 2.0, -2.0 , 2.0, levels=np.arange(-100,100,5), points=geodesic)
