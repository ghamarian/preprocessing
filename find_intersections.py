import json
from shapely.geometry.polygon import Polygon
from pprint import pprint
from matplotlib import pyplot as plt
from shapely.strtree import STRtree


def find_borders(polygons):
    str_tree = STRtree(polygons)

    for polygon in polygons:
        potentials = str_tree.query(polygon)

        for potential in potentials:
            if potential is polygon:
                continue
            intersect = potential.intersection(polygon)
            # if not intersect.is_empty and hasattr(intersect, 'exterior'):
            if not intersect.is_empty:
                yield intersect.convex_hull


def draw_borders(intersection_list, name, originals=None):
    fig = plt.figure(1, figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)

    for polygon in originals:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='red', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

    for polygon in intersection_list:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

    ax.set_title(f'{name}')
    plt.show()


if __name__ == '__main__':

    image_names = [
        'Set3_Image_GoldenUnit_coaxial.json',
        'Set3_Image_GoldenUnit_outerring.json',
        'Set4_Image_GoodSample_outerring.json',
        'Set4_Image_GoodSample_coaxial.json',
        'Set4_Image_Unit1_coaxial.json',
        'Set4_Image_Unit1_outerring.json',
        'Set5_Image_GoodSample_coaxial.json',
        'Set5_Image_GoodSample_outerring.json',

        # crossing wires as 1 poly
        'Set6_Unit3_WTRL0056_buffer00.json',
        'Set6_Unit3_WTRL0056_buffer01.json',
        'Set6_Unit4_WTRL0054_buffer00.json',
        'Set6_Unit4_WTRL0054_buffer01.json',
        # crossing wires as 1 poly
        'Set7_PBGA-Template-CEA_WTRL0049_buffer00.json',
        'Set7_PBGA-Template-CEA_WTRL0049_buffer01.json',
        'Set7_PBGA-Template-CEA_WTRL0051_buffer00.json',
        'Set7_PBGA-Template-CEA_WTRL0051_buffer01.json']

    for name in image_names:
        with open(f'./data/{name}') as data:
            polygons_json = json.load(data)

        try:
            polygon_points = [p['points'] for p in polygons_json['shapes'] if len(p['points']) > 1]
            polygons = [Polygon(points) for points in polygon_points if Polygon(points).is_valid]
            borders = find_borders(polygons)
            draw_borders(borders, name, originals=polygons)
        except Exception as e:
            print(f'{name} had an error with {e}')
