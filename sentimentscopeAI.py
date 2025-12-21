import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_predictive_star(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

    num_star = prediction + 1
    return num_star

text = "To be honest, I was bit disappointed with the service, but the food was okay."

print(f"The prediction for the review \"{text}\" is {get_predictive_star(text)}.")