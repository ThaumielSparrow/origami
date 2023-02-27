from origami_utils import Origami_Pretrained

model_name = "origami_stable"
model = Origami_Pretrained(model_name)


sequences = [""]
model.evaluate_sequences(model_chunk_size=128, savename="ARGN")