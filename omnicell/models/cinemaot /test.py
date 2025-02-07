
import pertpy as pt
from omnicell.models.cinemaot.predictor import CinemaOTPredictorWrapper

adata = pt.dt.cinemaot_example()

config = {
    "pert_key": "pert",
    "control": "ctrl",
    "thres": 0.5,
    "preweight_label": "cell_type0528"
}
model = CinemaOTPredictorWrapper(device="cpu", model_config=config)

model.train(adata)

effects = model.make_predict(adata, pert_id="IFNb")
print(effects.shape)  
