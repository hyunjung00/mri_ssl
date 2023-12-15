import json 
import os

'''

data_path = "../../../../data/crossmoda/ceT1/"

cet1_files = os.listdir(data_path)
label_list = []
mri_list = []

for ele in cet1_files:
    if 'Label' in ele: 
        label_list.append(ele)
    else:
        mri_list.append(ele)

print(len(label_list))
print(len(mri_list))

label_list = sorted(label_list)
mri_list = sorted(mri_list)


cet1_list = []
for i, _ in enumerate(label_list): 
    d = {}
    d["image"] = mri_list[i]
    d["label"] = label_list[i]
    
    cet1_list.append(d)


jsonString = json.dumps(cet1_list)
jsonFile = open("../jsons/ceT1.json", "w")
jsonFile.write(jsonString)
jsonFile.close()
'''

'''
'
data_path = "../../../../data/crossmoda/hrT2/"

hrT2_files = os.listdir(data_path)

hrT2_json_list = []
for i, _ in enumerate(hrT2_files): 
    d = {}
    d["image"] = hrT2_files[i]
    
    hrT2_json_list.append(d)


jsonString = json.dumps(hrT2_json_list)
jsonFile = open("../jsons/hrT2.json", "w")
jsonFile.write(jsonString)
jsonFile.close()
'''

data_path = "../../../../data/crossmoda/pseudo_hrT2/"

hrT2_files = os.listdir(data_path)

hrT2_val_list = []
for i, _ in enumerate(hrT2_files): 
    d = {}
    d["image"] = hrT2_files[i]
    
    hrT2_val_list.append(d)

jsonString = json.dumps(hrT2_val_list)
jsonFile = open("../jsons/pseduo_hrT2.json", "w")
jsonFile.write(jsonString)
jsonFile.close()