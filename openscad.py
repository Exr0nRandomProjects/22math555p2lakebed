# make sure to python3 -m pip install solidpython

from solid import scad_render_to_file
from solid import cube, sphere
from solid.utils import difference

from solid import translate
from solid.splines import catmull_rom_polygon, bezier_polygon
from euclid3 import Point2

points = [ Point2(0,0), Point2(1,1), Point2(2,1), Point2(2,-1) ]
shape = catmull_rom_polygon(points, show_controls=True)

bezier_shape = translate([3,0,0])(bezier_polygon(points, subdivisions=20))

# scad = difference()(
#     cube(10),  # Note the comma between each element!
#     sphere(10)
# ) + cube(3)

scad = bezier_shape

scad_render_to_file(scad, "out.scad")

