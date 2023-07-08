import json


d = {}

count = 1
with open('png/filelist.txt', 'r') as filelist: 
    for line in filelist:
        name = line.split('/')[0]
        if name not in d:
            d[name] = count
            count += 1

# Write the dictionary to a file
with open('labels.json', 'w') as f:
    json.dump(d, f, indent=4)
