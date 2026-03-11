import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

polygons = []

tree = ET.parse("data/2.xml")
root = tree.getroot()

mpp = float(root.attrib.get("MicronsPerPixel", "0"))


for annot in root.findall(".//Annotation"):
    for region in annot.findall(".//Region"):
        polygon = []
        for v in region.findall(".//Vertex"):
            x = float(v.get("X"))
            y = float(v.get("Y"))
            polygon.append((x, y))
        polygons.append(polygon)

plt.figure(figsize=(6,6))

all_y = [p[1] for poly in polygons for p in poly]
max_y = max(all_y)

for polygon in polygons:
    x, y = zip(*polygon)

    x = list(x) + [x[0]]
    y = [max_y - yi for yi in list(y) + [y[0]]]

    plt.plot(x, y, 'r-', linewidth=2)
    plt.fill(x, y, alpha=0.3, color = 'red')

plt.gca().set_aspect('equal')
plt.show()