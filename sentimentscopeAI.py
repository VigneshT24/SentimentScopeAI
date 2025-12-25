import torch
import json
import os
import textwrap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentScopeAI:
    ## Private attributes
    __model_name = None
    __tokenizer = None
    __model = None
    __json_file_path = None

    def __init__(self, file_path):
        """"Initialize the SentimentScopeAI class with the specified JSON file path."""
        self.__model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        self.__model = AutoModelForSequenceClassification.from_pretrained(self.__model_name)
        base_dir = os.path.dirname(__file__)
        self.__json_file_path = os.path.join(base_dir, file_path)


    def get_predictive_star(self, text):
        """"
            Predict the sentiment star rating for the given text review.

            Args:
                text (str): The text review to analyze.
            Returns:
                int: The predicted star rating (1 to 5).
        """
        inputs = self.__tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.__model(**inputs)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

        num_star = prediction + 1
        return num_star
    
    def output_all_reviews(self):
        """"
            Output all reviews from the JSON file in a formatted manner.

            Args:
                None
            Returns:
                None
        """
        with open(self.__json_file_path, 'r') as file:
            company_reviews = json.load(file)
            for i, entry in enumerate(company_reviews, 1):
                print(f"Review #{i}")
                print(f"Company Name: {entry['company_name']}")
                print(f"Service Name: {entry['service_name']}")
                print(f"Review: {textwrap.fill(entry['review'], width=70)}")
                print("\n\n")

    def calculate_all_review(self):
        pass