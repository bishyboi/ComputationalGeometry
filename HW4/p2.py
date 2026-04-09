import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set_theme(style='whitegrid')

num_points = 20
np.random.seed(0)
x_orig = np.random.randint(low=0, high=50, size=num_points)
y_orig = np.random.randint(low=0, high=50, size=num_points)

# Figure 1: Original Points
plt.figure()
sns.scatterplot(x=x_orig, y=y_orig)
plt.title("Original Points")

# Convex Hull
convex_hull = sp.spatial.ConvexHull(np.column_stack((x_orig, y_orig)))
hull_indices = convex_hull.vertices

# Figure 2: Convex Hull
plt.figure()
sns.scatterplot(x=x_orig, y=y_orig)
sns.scatterplot(x=x_orig[hull_indices], y=y_orig[hull_indices], color='red')
plt.title("Convex Hull Points")

x = x_orig[hull_indices]
y = y_orig[hull_indices]
N_pts = len(x)

# Lifting Points onto Paraboloid
z = x**2 + y**2

# Figure 3: Lifted Paraboloid
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(x, y, z)
plt.title("Lifted Points onto Paraboloid")

convex_3d = sp.spatial.ConvexHull(points=np.column_stack((x, y, z)))

k = [0, 0, 1]  # k-hat, or the z basis vector
valid = np.empty(len(convex_3d.equations), dtype=bool)

for i, eq in enumerate(convex_3d.equations):
    if np.dot(k, eq[:-1]) > 0:
        valid[i] = True
    else:
        valid[i] = False

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.adjacency = []
        self.simplices = []
        self.circumcenters = []

points = []
for x_i, y_i in zip(x, y):
    points.append(Point(x_i, y_i))

def plot_simplex(x, y, simplex, ax=None):
    if ax is None:
        ax = plt.gca()
    line1 = [simplex[0], simplex[1]]
    line2 = [simplex[1], simplex[2]]
    line3 = [simplex[0], simplex[2]]
    
    for line in [line1, line2, line3]:
        ax.plot(x[line], y[line], color='blue')

def set_adjacency(points, simplex):
    s = simplex.copy()
    shift = 1
    for row in simplex:
        neighbors = np.roll(s, shift)[1:]
        for n in neighbors:
            if n not in points[row].adjacency:
                points[row].adjacency.append(n)
        shift += 1

def get_circumcenter(x, y, simplex):
    x1, x2, x3 = x[simplex[0]], x[simplex[1]], x[simplex[2]]
    y1, y2, y3 = y[simplex[0]], y[simplex[1]], y[simplex[2]]
    
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    a_sq = x1**2 + y1**2
    b_sq = x2**2 + y2**2
    c_sq = x3**2 + y3**2

    u = (a_sq * (y2 - y3) + b_sq * (y3 - y1) + c_sq * (y1 - y2)) / d
    v = (a_sq * (x3 - x2) + b_sq * (x1 - x3) + c_sq * (x2 - x1)) / d
    
    return u, v

valid_simplices = []
for i, simplex in enumerate(convex_3d.simplices):
    if valid[i]:
        valid_simplices.append(simplex)
        u, v = get_circumcenter(x, y, simplex)
        set_adjacency(points, simplex)
        for row in simplex:
            points[row].simplices.append(simplex)
            points[row].circumcenters.append((u, v))

# Figure 4: Farthest-Point Delaunay Triangulation
fig, ax = plt.subplots()
for simplex in valid_simplices:
    plot_simplex(x, y, simplex, ax)
ax.scatter(x, y, color='red')
plt.title("Farthest-Point Delaunay Triangulation")


# Figure 5: Farthest-Point Voronoi Diagram (lines only)
fig, ax = plt.subplots()
ax.scatter(x, y, color='red')

cc_list = [get_circumcenter(x, y, s) for s in valid_simplices]

for i in range(len(valid_simplices)):
    for j in range(i + 1, len(valid_simplices)):
        shared_vertices = set(valid_simplices[i]).intersection(set(valid_simplices[j]))
        if len(shared_vertices) == 2:
            cc1 = cc_list[i]
            cc2 = cc_list[j]
            ax.plot([cc1[0], cc2[0]], [cc1[1], cc2[1]], color='green')

edge_counts = {}
for i, simplex in enumerate(valid_simplices):
    edges = [
        tuple(sorted([simplex[0], simplex[1]])),
        tuple(sorted([simplex[1], simplex[2]])),
        tuple(sorted([simplex[0], simplex[2]]))
    ]
    for edge in edges:
        if edge not in edge_counts:
            edge_counts[edge] = []
        edge_counts[edge].append(i)

rays_for_point = {i: [] for i in range(N_pts)}
center_x = np.mean(x)
center_y = np.mean(y)

for edge, simplices_indices in edge_counts.items():
    if len(simplices_indices) == 1:
        simplex_idx = simplices_indices[0]
        simplex = valid_simplices[simplex_idx]
        cc = cc_list[simplex_idx]
        
        A, B = edge
        C = next(v for v in simplex if v != A and v != B)
        
        dx = x[B] - x[A]
        dy = y[B] - y[A]
        
        N_vec = np.array([-dy, dx], dtype=float)
        AC = np.array([x[C] - x[A], y[C] - y[A]], dtype=float)
        
        if np.dot(N_vec, AC) < 0:
            N_vec = -N_vec
            
        norm = np.linalg.norm(N_vec)
        if norm > 0:
            N_vec = N_vec / norm
            # Add a point very far away
            ray_end = (cc[0] + N_vec[0] * 5000, cc[1] + N_vec[1] * 5000)
            ax.plot([cc[0], ray_end[0]], [cc[1], ray_end[1]], color='green', linestyle='dashed')
            rays_for_point[A].append(ray_end)
            rays_for_point[B].append(ray_end)

ax.set_xlim(min(x) - 10, max(x) + 10)
ax.set_ylim(min(y) - 10, max(y) + 10)
plt.title("Farthest-Point Voronoi Diagram")


# Figure 6: Colored Farthest-Point Voronoi Diagram (Clean view)
fig, ax = plt.subplots()
colors = sns.color_palette("hls", N_pts)

for i in range(N_pts):
    # Combine the true Voronoi vertices and ray extensions to infinity
    cell_verts = points[i].circumcenters + rays_for_point[i]
    if len(cell_verts) >= 3:
        # Sort vertices counterclockwise around their centroid
        cx = sum(v[0] for v in cell_verts) / len(cell_verts)
        cy = sum(v[1] for v in cell_verts) / len(cell_verts)
        cell_verts.sort(key=lambda v: math.atan2(v[1] - cy, v[0] - cx))
        
        poly = plt.Polygon(cell_verts, facecolor=colors[i], edgecolor='black', alpha=0.5)
        ax.add_patch(poly)
    
    # Plot the point itself with the same color
    ax.scatter(x[i], y[i], color=colors[i], edgecolor='black', zorder=5)

ax.set_xlim(min(x) - 10, max(x) + 10)
ax.set_ylim(min(y) - 10, max(y) + 10)
plt.title("Colored Clean Farthest-Point Voronoi Diagram")

plt.show()
