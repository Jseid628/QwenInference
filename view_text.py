import json

# test: load test.txt and print a few entries
test_labels_path = "./data/test.txt"

with open(test_labels_path, "r") as f:
    # f.readlines() returns a list
    lines = f.readlines()

# Preview first 5 lines
for line in lines[:5]:
    print(line.strip())
    
label_dict = {}
for line in lines:
    parts = line.strip().split()
    fname = parts[1]
    label = parts[2]
    label_dict[fname] = label

with open("test_labels_dict.json", "w") as f:
    json.dump(label_dict, f)

