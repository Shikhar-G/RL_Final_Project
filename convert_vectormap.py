import os
import json

file = open("CDL_Ground.vectormap.txt", "r")
json_obj = []
for line in file:
    if len(line) > 0:
        vals = line.split(",")
        json_obj.append(
            {
                "p0": {"x": float(vals[0]), "y": float(vals[1])},
                "p1": {"x": float(vals[2]), "y": float(vals[3])},
            }
        )
file.close()
json.dump(json_obj, open("CDL_Ground.vectormap.json", "w"), indent=4)
