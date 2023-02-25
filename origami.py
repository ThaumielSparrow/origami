import torch
import esm
from esm.esmfold.v1.esmfold import ESMFold


model_name = "esmfold_3B_v1"
url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"

model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
cfg = model_data["cfg"]["model"]
model_state = model_data["model"]

model = ESMFold(esmfold_config=cfg)
expected_keys = set(model.state_dict().keys())
found_keys = set(model_state.keys())

missing_essential_keys = []
for missing_key in expected_keys - found_keys:
    if not missing_key.startswith("esm."):
        missing_essential_keys.append(missing_key)

if missing_essential_keys:
    raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

model.load_state_dict(model_state, strict=False)


model = model.eval().cuda()
model.set_chunk_size(128)

sequence = input("Enter FASTA sequence: ")

with torch.no_grad():
    output = model.infer_pdb(sequence)

with open("result.pdb", "w") as f:
    f.write(output)

print("Fold prediction complete.")
