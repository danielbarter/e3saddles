from e3saddles.configuration_space import *

@jax.jit
def muller_brown(point):
    x = point[0]
    y = point[1]

    ai = [-200.0, -100.0, -170.0, 15.0]
    bi = [-1.0, -1.0, -6.5, 0.7]
    ci = [0.0, 0.0, 11.0, 0.6]
    di = [-10.0, -10.0, -6.5, 0.7]

    xi = [1.0, 0.0, -0.5, -1.0]
    yi = [0.0, 0.5, 1.5, 1.0]

    total = 0.0
    for i in range(4):
        total += ai[i] * jnp.exp(bi[i] * (x - xi[i]) * (x - xi[i]) + ci[i] * (x - xi[i]) * (y - yi[i]) + di[i] * (y - yi[i]) * (y - yi[i]))


    return  total


minima_1 = find_minima(muller_brown, jnp.array([-0.7, 1.5]), 50000, 0.0001)
minima_2 = find_minima(muller_brown, jnp.array([0.0, 0.5]),  50000, 0.0001)
minima_3 = find_minima(muller_brown, jnp.array([0.5, 0.0]),  50000, 0.0001)

initial_points = compute_initial_points(minima_1, minima_2, 30)

geodesic=find_geodesic(muller_brown, initial_points, minima_1, minima_2, 20000, 0.000001, 100)
contour_2d(muller_brown, -1.7, 1.3, -0.5 , 2.2, levels=np.arange(-200,200,10), points=geodesic)
