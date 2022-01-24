from e3saddles.configuration_space import *

c = ConfigurationSpace(3)
f = c.random_surface(42, 10, 5, 20.0)

minima_1 = find_minima(f, c.random_point(1), 100000, 0.01)
minima_2 = find_minima(f, c.random_point(2), 100000, 0.01)
initial_points = compute_initial_points(minima_1, minima_2, 200)
find_geodesic(f, initial_points, minima_1, minima_2, 50000, 0.01)
