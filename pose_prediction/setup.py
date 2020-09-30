import json
import importlib

def load_model(fullpath_json, device):
    with open(fullpath_json) as json_file:
        settings = json.load(json_file)
        import_completion = settings["import_completion"]
        import_prediction = settings["import_prediction"]
    completion = importlib.import_module(import_completion).get_pose_completion(device)
    prediction = importlib.import_module(import_prediction).get_pose_prediction(device)
    return completion, prediction


  
