import torch
import esm
from esm.esmfold.v1.esmfold import ESMFold

class Origami_Pretrained:
    """
    Class to load a pretrained Origami model into memory. Will only work with Origami (and some ESM) models.
    """
    def __init__(self, model="origami_stable"):
        if model == "origami_stable":
            model_name = "esmfold_3B_v1"
        else:
            model_name = model
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        
        self.model_data = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
        self.cfg = self.model_data["cfg"]["model"]
        self.model_state = self.model_data["model"]

        self.model = ESMFold(esmfold_config=self.cfg)
        self.expected_keys = set(self.model.state_dict().keys())
        self.found_keys = set(self.model_state.keys())
        
        self.model.load_state_dict(self.model_state, strict=False)

    def check_keys(self):
        missing_essential_keys = []
        for missing_key in self.expected_keys - self.found_keys:
            if not missing_key.startswith("esm."):
                missing_essential_keys.append(missing_key)
        
        if (missing_essential_keys):
            raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")
    

    def evaluate_sequences(self, sequences=[], model_chunk_size=0, save=True, savename="result"):
        self.check_keys()
        model = self.model.eval().cuda()
        if (model_chunk_size>0):
            model.set_chunk_size(model_chunk_size)
        
        if not sequences:
            sequences = input("No sequences provided. Enter FASTA sequence(s), separated by comma: ")
            sequences_arr = sequences.replace(" ", "").split(',')
        else:
            sequences_arr = sequences
        
        for idx, i in enumerate(sequences_arr):
            with torch.no_grad():
                output = model.infer_pdb(i)
            
            if (save):
                with open(f"{savename}-{idx}.pdb", "w") as f:
                    f.write(output)
        
        print("Fold prediction complete")
