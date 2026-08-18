import os
import glob
import json
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


arr1 = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
arr2 = np.array([[51, 61, 91], [121, 118, 127]])

# Creating a Python Dictionary
jsData = {
    "i": 25,
    "j": 4,
    "arr1": arr1,
    "arr2": arr2
}

print(jsData)
print(jsData["i"])


#### Serialization ####
encodedNumpyData = json.dumps(jsData, cls=NumpyArrayEncoder)  # use dump() to write array into file
print("Printing JSON serialized NumPy array")
print(encodedNumpyData)

#### Deserialization ####
decodedArrays = json.loads(encodedNumpyData)


# Writing to a file
# **IMPORTANT:**
# - Saving the file does not need the serialization az it is done via cls=NumpyArrayEncoder
with open("./inputs_to_json/inputs.json", "w") as write_file:
    json.dump(jsData, write_file, cls=NumpyArrayEncoder)


# ### Reading from the File
with open("./inputs_to_json/inputs.json", "r") as read_file:
    decodedArray = json.load(read_file)

print(decodedArray)
print(decodedArray["j"])


# ## IMPORTANTTTTTTTTT
arr1_retrieved = np.asarray(decodedArray["arr1"])
print(arr1_retrieved)

# ### Iterate over Python Dictionary
for key in jsData:
    print(key)

for key in jsData:
    print(key, '->', jsData[key])

for key, value in jsData.items():
    print(key, '->', value)


# ### Importing the Json and then Assign the values of eacch key to a parameter with the name of that key
with open("./inputs_to_json/inputs.json", "r") as read_file:
    inputs = json.load(read_file)

# Assign the value of each Json's key to itself.
for key in inputs:
    if isinstance(inputs[key], list):
        globals()[key] = np.asarray(inputs[key])
    else:
        globals()[key] = inputs[key]


# ### Selct the most recent JSON file in the 'results' directory
# Selct the most recent JSON file in the 'results' directory
list_of_files = glob.glob('./results/*.json')
latest_json_file = max(list_of_files, key=os.path.getctime)
print(latest_json_file)

with open(latest_json_file, "r") as read_file:
    output = json.load(read_file)

output.keys()


# #### Get just the filename of the most recent file
latest_json_file_filename_with_extention = os.path.basename(latest_json_file)

latest_json_file_filename_without_extention = os.path.splitext(latest_json_file_filename_with_extention)[0]
latest_json_file_filename_without_extention
