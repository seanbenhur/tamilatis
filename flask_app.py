from waitress import serve
import json
from sklearn.preprocessing import LabelEncoder
import flask
from flask import Flask, jsonify, request 
from tamilatis.predict import TamilATISPredictor 
from tamilatis.model import JointATISModel
import numpy as np
import time 

app = Flask(__name__)


model_name = "microsoft/xlm-align-base"
tokenizer_name = "microsoft/xlm-align-base"
num_labels = 78
num_intents = 23
checkpoint_path = "models/xlm_align_base.bin"
intent_encoder_path = "models/intent_classes.npy"
ner_encoder_path = "models/ner_classes.npy"


def prediction_from_model(text):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(ner_encoder_path)
  
    intent_encoder = LabelEncoder()
    intent_encoder.classes_ = np.load(intent_encoder_path)
  
    model  = JointATISModel(model_name,num_labels,num_intents)
    predictor = TamilATISPredictor(model,checkpoint_path,tokenizer_name,
                              label_encoder,intent_encoder,num_labels)
    slot_prediction, intent_preds = predictor.get_predictions(text)
    return slot_prediction, intent_preds

@app.route('/', methods=['POST'])
def predict():
    #get request
    data = request.get_json()
    #get outputs from the model
    start_time = time.perf_counter()
    slot_prediction, intent_preds = prediction_from_model(data)
    #for calculating time
    end_time = time.perf_counter()
    output_data = {
        "time_spent" : end_time - start_time,
        "slot_prediction": slot_prediction.tolist(),
        "intent_preds": [intent_preds]
    }

    return flask.Response(response=json.dumps(output_data))

if __name__ == "__main__":
    serve(app=app, port=7000)
   


