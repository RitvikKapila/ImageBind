from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import json

f = open('labelname_to_labeldescription_mapping.json')

label_data = json.load(f)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

imagebind_embeddings = {}
for dataset, label_dict in label_data.items():
    # print(dataset, label_dict)
    imagebind_embeddings[dataset] = {}

    label_num_list = list(label_data[dataset].keys())
    label_text_list = list(label_data[dataset].values())
    print(label_num_list, label_text_list)


    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(label_text_list, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)

    # imagebind_embeddings[dataset][label_num] = embeddings[ModalityType.TEXT]
    for i, num in enumerate(label_num_list):
        imagebind_embeddings[dataset][num] = embeddings[ModalityType.TEXT][i].tolist()

print(imagebind_embeddings)

with open("labeldescription_embeddings_imagebind.json", "w") as write_file:
    json.dump(imagebind_embeddings, write_file, indent=4)

f.close()