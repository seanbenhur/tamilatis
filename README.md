<h1 align="center">
  <br />
  MultiTask Learning in Intent Detection and Slot Prediction for Tamil Conversational Dialogues using Multilingual Pretrained Models
</h1>



The aim of this project is to develop an intent detction and slot prediction system for Tamil language. An open source dataset from the paper "TamilATIS: Dataset for Task-Oriented Dialog in Tamil (S et al., DravidianLangTech 2022)" was used.Both the Single Task learning based approach and Multi-task learning approaches are experimented. In MultiTask Learning, we use a [Random Loss Weighting](https://arxiv.org/abs/2111.10603) for intent-detection to account for class-imbalance.

## Getting Started

### Dataset

The TamilATIS dataset was created by translating the ATIS dataset to English and then the slots are annotated manually.
To get the dataset, email the authors of the paper "TamilATIS: Dataset for Task-Oriented Dialog in Tamil (S et al., DravidianLangTech 2022)".


### Demo

The demo for this project is available [here](https://huggingface.co/spaces/seanbenhur/tamilatis)

### Usage

As a first step, clone this repo. 

To run the MultiTask Learning experiements, Edit all the configurations in the configs folder.This project uses Hydra as a configuration management tool. If you are new to hydra, I would recommend this [tutorial](https://www.ravirajag.dev/blog/mlops-hydra-config).

After setting up all the configs, 


```bash
python3 main.py
```

The single task learning experiments are inside, the ```stl``` folder.
## Results

To experiment with this dataset, two multilingual pretrained models are used XLM-Roberta-Base and XLM-Align-Base. Both the Single task learning and Multi-task learning approach is used.

### Single Task Learning

#### Intent-Detection

|     **Model**    | **Accuracy** | **Macro F1** | **Weighted F1** | 
|:----------------:|:------------:|:------------:|:-----------:|
| XLM-ROBERTA-BASE             |    0.9252   |    0.4080   |    0.9166 | 
| XLM-ALIGN-BASE          |  0.9013    | 0.2506    |  0.8839   |  


#### Slot Filling

|     **Model**    | **Accuracy** | **Macro F1** | **Weighted F1** | 
|:----------------:|:------------:|:------------:|:-----------:|
| XLM-ROBERTA-BASE             |    0.9507  |    0.6298    |    0.9205  | 
| XLM-ALIGN-BASE          |    0.9428   | 0.57       | 0.89      |  



### Multi Task Learning

#### Intent-Detection

|     **Model**    | **Accuracy** | **Macro F1** | **Weighted F1** | 
|:----------------:|:------------:|:------------:|:-----------:|
| XLM-ROBERTA-BASE    |    0.9476  |    0.6458    |    0.9434   |          
| XLM-ALIGN-BASE          |    0.9566    |    0.5828   |    0.9521  |  


#### Slot Filling

|     **Model**    | **Accuracy** | **Macro F1** | **Weighted F1** | 
|:----------------:|:------------:|:------------:|:-----------:|
| XLM-ROBERTA-BASE  |       0.9600       |      0.7508     |   0.9432    | 
| XLM-ALIGN-BASE          |    0.9243    |    0.7863   |    0.9395   |  


## Findings

* The RLW didn't work when used with span loss, The run can be found be found [here](https://wandb.ai/seanbenhur/tamilatis/runs/9q67z8xl?workspace=user-seanbenhur), so the results were not included here.
* XLM large and other large model didn't work



## Acknowledgements

### TamilATIS Dataset


**Citation**

Ramaneswaran S, Sanchit Vijay, and Kathiravan Srinivasan. 2022. TamilATIS: Dataset for Task-Oriented Dialog in Tamil. In Proceedings of the Second Workshop on Speech and Language Technologies for Dravidian Languages, pages 25–32, Dublin, Ireland. Association for Computational Linguistics.


**Bibtex**

```tex
@inproceedings{s-etal-2022-tamilatis,
    title = "{T}amil{ATIS}: Dataset for Task-Oriented Dialog in {T}amil",
    author = "S, Ramaneswaran  and
      Vijay, Sanchit  and
      Srinivasan, Kathiravan",
    booktitle = "Proceedings of the Second Workshop on Speech and Language Technologies for Dravidian Languages",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.dravidianlangtech-1.4",
    doi = "10.18653/v1/2022.dravidianlangtech-1.4",
    pages = "25--32"
}
```