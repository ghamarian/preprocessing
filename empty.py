from matplotlib import pyplot as plt
from descartes import PolygonPatch
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon

# polygon = Polygon([(0, 0), (1, 1), (1, 0)])
# polygon1 = Polygon([(5.5, 5.5), (6.5, 6.5), (6.5, 5.5)])
# k = polygon1.intersection(polygon)
# x, y = k.exterior.xy
# x0, y0 = polygon.exterior.xy
# x1, y1 = polygon1.exterior.xy
#
fig = plt.figure(1, figsize=(5, 5), dpi=90)
ax = fig.add_subplot(111)

# ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
# ax.plot(x0, y0, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
# ax.plot(x1, y1, color='black', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
# ax.set_title('Polygon')
# plt.show()

from shapely.geometry import Polygon
from shapely.strtree import STRtree

polys = [Polygon(((0, 0), (1, 0), (1, 1))), Polygon(((0, 1), (0, 0), (1, 0))), Polygon(((10, 10), (11, 10), (11, 11)))]
s = STRtree(polys)
query_geom = Polygon(((-1, -1), (2, 0), (2, 2), (-1, 2)))
result = s.query(polys[2])
# polys[0] in result

for p in polys:
    x, y = p.exterior.xy
    ax.plot(x, y, color='blue', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

for p in result:
    x, y = p.exterior.xy
    ax.plot(x, y, color='red', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

ax.set_title('Polygon')
plt.show()
