from model import JointATISModel
from dataset import ATISDataset
from predict_onnx import TamilATISPredictor
import joblib

from sklearn.preprocessing import LabelEncoder
import torch 
from transformers import AutoTokenizer

model_path = "/content/tamil_atis/pytorch_model.bin"
model_name = "xlm-roberta-base"
num_labels = 78
num_intents = 23
label_encoder_path = "/content/drive/MyDrive/models/tamilatis/label_encoder.joblib"
intent_encoder_path = "/content/drive/MyDrive/models/tamilatis/intent_encoder.joblib"

model = JointATISModel(model_name,num_labels,num_intents)
#model.load_state_dict(torch.load(model_path))



label_encoder = joblib.load(label_encoder_path)
intent_encoder = joblib.load(intent_encoder_path)
 
text =  "எனக்கு டெல்லியில் இருந்து சென்னைக்கு விமானம் வேண்டும்"
predictor = TamilATISPredictor(model,model_path,model_name, label_encoder,intent_encoder,num_labels)
intents, ner_outputs = predictor.get_predictions(text)
print(ner_outputs)
print(intents)