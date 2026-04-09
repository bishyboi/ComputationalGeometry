import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_code(src):
    lines = [line + "\n" for line in src.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = lines[-1].strip("\n")  # remove trailing newline from last line
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

def add_md(src):
    lines = [line + "\n" for line in src.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = lines[-1].strip("\n")
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

add_code("""import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set_theme(style='whitegrid')""")

add_code("""num_points = 20
np.random.seed(0)
x_orig = np.random.randint(low=0, high=50, size=num_points)
y_orig = np.random.randint(low=0, high=50, size=num_points)

# Figure 1: Original Points
plt.figure()
sns.scatterplot(x=x_orig, y=y_orig)
plt.title("Original Points")
plt.show()""")

add_md("Only points on the convex hull can be included on the voronoi diagram")

add_code("""# Convex Hull
convex_hull = sp.spatial.ConvexHull(np.column_stack((x_orig, y_orig)))
hull_indices = convex_hull.vertices

# Figure 2: Convex Hull
plt.figure()
sns.scatterplot(x=x_orig, y=y_orig)
sns.scatterplot(x=x_orig[hull_indices], y=y_orig[hull_indices], color='red')
plt.title("Convex Hull Points")
plt.show()""")

add_code("""x = x_orig[hull_indices]
y = y_orig[hull_indices]
N_pts = len(x)""")

add_md("# Lifting Points onto Paraboloid")

add_code("""z = x**2 + y**2
z""")

add_code("""# Figure 3: Lifted Paraboloid
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(x, y, z)
plt.title("Lifted Points onto Paraboloid")
plt.show()""")

add_code("convex_3d = sp.spatial.ConvexHull(points=np.column_stack((x, y, z)))")

add_md("Equations are the planes of the convex hull described as the components of the normal vector + the offset, or \n\n$ \\vec{n} + b $")

add_code("convex_3d.equations[0]")

add_md("We can determine which faces are pointing upwards by only keeping the faces that have \n\n$ \\vec{n} \\cdot \\hat{k} > 0 $")

add_code("""k = [0, 0, 1]  # k-hat, or the z basis vector
valid = np.empty(len(convex_3d.equations), dtype=bool)

for i, eq in enumerate(convex_3d.equations):
    if np.dot(k, eq[:-1]) > 0:
        valid[i] = True
    else:
        valid[i] = False""")

add_md("# Creating a class to organize adjacency between convex hull faces (simplices), circumcenters, and points before Farthest-Delaunay Triangulation Projection")

add_code("""class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.adjacency = []
        self.simplices = []
        self.circumcenters = []""")

add_code("""points = []
for x_i, y_i in zip(x, y):
    points.append(Point(x_i, y_i))""")

add_code("""def plot_simplex(x, y, simplex, ax=None):
    if ax is None:
        ax = plt.gca()
    line1 = [simplex[0], simplex[1]]
    line2 = [simplex[1], simplex[2]]
    line3 = [simplex[0], simplex[2]]
    
    for line in [line1, line2, line3]:
        ax.plot(x[line], y[line], color='blue')""")

add_code("""def set_adjacency(points, simplex):
    s = simplex.copy()
    shift = 1
    for row in simplex:
        neighbors = np.roll(s, shift)[1:]
        for n in neighbors:
            if n not in points[row].adjacency:
                points[row].adjacency.append(n)
        shift += 1""")

add_code("""def get_circumcenter(x, y, simplex):
    x1, x2, x3 = x[simplex[0]], x[simplex[1]], x[simplex[2]]
    y1, y2, y3 = y[simplex[0]], y[simplex[1]], y[simplex[2]]
    
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    a_sq = x1**2 + y1**2
    b_sq = x2**2 + y2**2
    c_sq = x3**2 + y3**2

    u = (a_sq * (y2 - y3) + b_sq * (y3 - y1) + c_sq * (y1 - y2)) / d
    v = (a_sq * (x3 - x2) + b_sq * (x1 - x3) + c_sq * (x2 - x1)) / d
    
    return u, v""")

add_code("""valid_simplices = []
for i, simplex in enumerate(convex_3d.simplices):
    if valid[i]:
        valid_simplices.append(simplex)
        u, v = get_circumcenter(x, y, simplex)
        set_adjacency(points, simplex)
        for row in simplex:
            points[row].simplices.append(simplex)
            points[row].circumcenters.append((u, v))""")

add_code("""# Figure 4: Farthest-Point Delaunay Triangulation
fig, ax = plt.subplots()
for simplex in valid_simplices:
    plot_simplex(x, y, simplex, ax)
ax.scatter(x, y, color='red')
plt.title("Farthest-Point Delaunay Triangulation")
plt.show()""")

add_md("To create the farthest-delaunay voronoi diagram, you simply find the connected components of the circumcenters of the farthest-delaunay triangulation.")

add_code("""# Figure 5: Farthest-Point Voronoi Diagram (lines only)
fig, ax = plt.subplots()
ax.scatter(x, y, color='red')

# 1. To form the Farthest-Point Voronoi Diagram, the vertices of the cells are exactly 
# the circumcenters of the Farthest-Point Delaunay triangulation.
cc_list = [get_circumcenter(x, y, s) for s in valid_simplices]

# 2. Draw Bounded Voronoi Edges: 
# If two triangles in the Delaunay triangulation share an edge (exactly 2 vertices),
# their corresponding circumcenters are connected by a bounded Voronoi edge.
for i in range(len(valid_simplices)):
    for j in range(i + 1, len(valid_simplices)):
        shared_vertices = set(valid_simplices[i]).intersection(set(valid_simplices[j]))
        if len(shared_vertices) == 2:
            cc1 = cc_list[i]
            cc2 = cc_list[j]
            ax.plot([cc1[0], cc2[0]], [cc1[1], cc2[1]], color='green')

# 3. Find Boundary Edges of the Triangulation:
# A boundary edge of the triangulation belongs to only one valid simplex.
# We map each edge to the indices of the simplices that contain it.
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

# rays_for_point will store the distant "infinity" points to later close the Voronoi cells 
rays_for_point = {i: [] for i in range(N_pts)}

# 4. Draw Unbounded Voronoi Edges (Rays):
# For edges on the boundary of the hull, the Voronoi edge extends to infinity.
# It is the perpendicular bisector of the boundary edge.
for edge, simplices_indices in edge_counts.items():
    if len(simplices_indices) == 1:
        # Get the single simplex containing this boundary edge
        simplex_idx = simplices_indices[0]
        simplex = valid_simplices[simplex_idx]
        
        # cc represents the origin of our ray (the simplex's circumcenter)
        cc = cc_list[simplex_idx]
        
        A, B = edge
        # Identify the third vertex C of the triangle to help orient our ray
        C = next(v for v in simplex if v != A and v != B)
        
        # Compute the direction vector of the edge AB
        dx = x[B] - x[A]
        dy = y[B] - y[A]
        
        # 5. Determine the Ray Direction (Inward Normal):
        # N_vec is a perpendicular vector to the edge AB.
        N_vec = np.array([-dy, dx], dtype=float)
        
        # We need the ray to point TOWARDS the interior of the polygon for a Farthest-Point Voronoi Diagram
        # (Unlike an ordinary Voronoi diagram where rays point outward).
        # We check the dot product of N_vec and the vector AC. If it is negative, N_vec is pointing
        # away from vertex C (outward), so we flip it to point inward.
        AC = np.array([x[C] - x[A], y[C] - y[A]], dtype=float)
        if np.dot(N_vec, AC) < 0:
            N_vec = -N_vec
            
        norm = np.linalg.norm(N_vec)
        if norm > 0:
            # Normalize the perpendicular vector
            N_vec = N_vec / norm
            
            # Extend the ray far into the distance (acting as our point at infinity)
            ray_end = (cc[0] + N_vec[0] * 5000, cc[1] + N_vec[1] * 5000)
            
            # Plot the dashed ray separating the respective farthest-point regions
            ax.plot([cc[0], ray_end[0]], [cc[1], ray_end[1]], color='green', linestyle='dashed')
            
            # Save the point at infinity for later when we color the regions of points A and B
            rays_for_point[A].append(ray_end)
            rays_for_point[B].append(ray_end)

ax.set_xlim(min(x) - 10, max(x) + 10)
ax.set_ylim(min(y) - 10, max(y) + 10)
plt.title("Farthest-Point Voronoi Diagram")
plt.show()""")

add_md("Final Plot: Farthest-Point Voronoi Diagram with Points and Corresponding Regions Colored Identically, Removing Unnecessary Lines")

add_code("""# Figure 6: Colored Farthest-Point Voronoi Diagram (Clean view)
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

plt.show()""")

with open('p2_re.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
