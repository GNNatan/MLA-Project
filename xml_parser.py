import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

polygon = []

tree = ET.parse("1.xml")
root = tree.getroot()

mpp = float(root.attrib.get("MicronsPerPixel", "0"))

for annot in root.findall(".//Annotation"):
    for region in annot.findall(".//Region"):
        for v in region.findall(".//Vertex"):
            x = float(v.get("X"))
            y = float(v.get("Y"))
            polygon.append((round(x), round(y)))


x, y = zip(*polygon)
x += (x[0], )
y += (y[0], )
y_flipped = [] 
for e in y:
    y_flipped.append(max(y)- e)
y = y_flipped
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'r-', linewidth=2)
plt.fill(x, y, alpha=0.3, color='red')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()