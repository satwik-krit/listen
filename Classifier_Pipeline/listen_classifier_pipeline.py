# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from joblib import load,dump

model_component = load("model_finaL.joblib")

model_valve = load("model_valve.joblib")
model_fan = load("model_fan.joblib")
model_pump = load("model_pump.joblib")
model_slider = load("model_slider.joblib")

component_models = {
    0: model_fan,
    1: model_pump,
    2: model_slider,
    3: model_valve
}

component_outputs = {
    0: {0: "fan 00", 1: "fan 02", 2: "fan 04", 3: "fan 06"},
    1: {0: "pump 00", 1: "pump 02", 2: "pump 04", 3: "pump 06"},
    2: {0: "slider 00", 1: "slider 02", 2: "slider 04", 3: "slider 06"},
    3: {0: "valve 00", 1: "valve 02", 2: "valve 04"}
}

to_search = np.array(""" your numpy array to be tested to be entered here """).reshape(1,8)
df = pd.DataFrame(to_search)

def predict(np_arr):
  print("0")
  y_comp = model_component.predict(np_arr) # Extract the integer value from the numpy array
  print("1")
  if y_comp == 0:
    y_fan = model_fan.predict(np_arr)
    return component_outputs[y_comp.item()][y_fan.item()]
  elif y_comp == 1:
    y_pump = model_pump.predict(np_arr)
    return component_outputs[y_comp.item()][y_pump.item()]
  elif y_comp == 2:
    y_slider = model_slider.predict(np_arr)
    return component_outputs[y_comp.item()][y_slider.item()]
  elif y_comp == 3:
    y_valve = model_valve.predict(np_arr)
    return component_outputs[y_comp.item()][y_valve.item()]

print(predict(df))

