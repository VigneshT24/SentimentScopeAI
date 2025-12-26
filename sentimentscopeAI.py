import torch
import json
import os
import string
import random
import textwrap
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5Tokenizer, set_seed)

class SentimentScopeAI:
    ## Private attributes
    __hf_model_name = None
    __hf_tokenizer = None
    __hf_model = None
    __pytorch_model_name = None
    __pytorch_tokenizer = None
    __pytorch_model = None
    __json_file_path = None
    __service_name = None
    __device = None

    def __init__(self, file_path):
        """Initialize the SentimentScopeAI class with the specified JSON file path."""
        self.__hf_model_name = "Vamsi/T5_Paraphrase_Paws"
        self.__pytorch_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        base_dir = os.path.dirname(__file__)
        self.__json_file_path = os.path.join(base_dir, file_path)
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def hf_model(self):
        """Lazy loader for the Paraphrase Model."""
        if self.__hf_model is None:
            print("Loading T5 Paraphrase Model for the first time...")
            self.__hf_model = AutoModelForSeq2SeqLM.from_pretrained(self.__hf_model_name)
        return self.__hf_model

    @property
    def hf_tokenizer(self):
        """Lazy loader for the T5 Tokenizer."""
        if self.__hf_tokenizer is None:
            self.__hf_tokenizer = T5Tokenizer.from_pretrained(self.__hf_model_name, legacy=True)
        return self.__hf_tokenizer

    @property
    def pytorch_tokenizer(self):
        """Lazy loader for the Pytorch Tokenizer."""
        if self.__pytorch_tokenizer is None:
            print(f"Loading BERT Tokenizer...")
            self.__pytorch_tokenizer = AutoTokenizer.from_pretrained(self.__pytorch_model_name)
        return self.__pytorch_tokenizer

    @property
    def pytorch_model(self):
        """Lazy loader for the Pytorch Model."""
        if self.__pytorch_model is None:
            print(f"Loading BERT Model onto {self.__device}...")
            self.__pytorch_model = AutoModelForSequenceClassification.from_pretrained(
                self.__pytorch_model_name
            ).to(self.__device)
        return self.__pytorch_model

    def __get_predictive_star(self, text) -> int:
        """
            Predict the sentiment star rating for the given text review.

            Args:
                text (str): The text review to analyze.
            Returns:
                int: The predicted star rating (1 to 5).
        """
        inputs = self.pytorch_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.pytorch_model(**inputs)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

        num_star = prediction + 1
        return num_star
    
    def output_all_reviews(self) -> None:
        """
            Output all reviews from the JSON file in a formatted manner.

            Args:
                None
            Returns:
                None
        """
        try:
            with open(self.__json_file_path, 'r') as file:
                company_reviews = json.load(file)
                for i, entry in enumerate(company_reviews, 1):
                    print(f"Review #{i}")
                    print(f"Company Name: {entry['company_name']}")
                    print(f"Service Name: {entry['service_name']}")
                    print(f"Review: {textwrap.fill(entry['review'], width=70)}")
                    print("\n\n")
        except FileNotFoundError:
            print("The JSON file you inputted doesn't exist. Please input a valid company review file.")
        except json.JSONDecodeError:
            print("Could not decode JSON file. Check for valid JSON syntax.")
        except PermissionError:
            print("Permission denied to open the JSON file.")
        except Exception as e:
            print(f"An unexpected error occured: {e}")

    def __calculate_all_review(self) -> int:
        """
            Calculate and print the predicted star ratings for all reviews in the JSON file.

            Args:
                None
            Returns:
                tuple: A tuple containing the total number of reviews and the average star rating.
        """
        try:
            with open(self.__json_file_path, 'r') as reviews_file:
                all_reviews = json.load(reviews_file)
                sum = 0
                num_reviews = 0
                for i, entry in enumerate(all_reviews, 1):
                    sum += self.__get_predictive_star(entry['review'])
                    self.__service_name = entry['service_name']
                    num_reviews = i
            return (sum / num_reviews) if num_reviews != 0 else 0
        except FileNotFoundError:
            print("The JSON file you inputted doesn't exist. Please input a valid company review file.")
        except json.JSONDecodeError:
            print("Could not decode JSON file. Check for valid JSON syntax.")
        except PermissionError:
            print("Permission denied to open the JSON file.")
        except Exception as e:
            print(f"An unexpected error occured: {e}")
    
    def __paraphrase_statement(self, statement: str) -> list[str]:
        """Generates multiple unique paraphrased variations of a given string.

        Uses a Hugging Face transformer model to generate five variations of the 
        input statement. Results are normalized (lowercased, stripped of 
        punctuation, and whitespace-cleaned) to ensure uniqueness.

        Args:
            statement (str): The text to be paraphrased.

        Returns:
            list[str]: A list of unique, cleaned paraphrased strings. 
                Returns [""] if the input is None, empty, or whitespace.
        """
        set_seed(random.randint(0, 2**32 - 1))
        
        if statement is None or statement.isspace() or statement == "":
            return [""]

        prompt = f"paraphrase: {statement}"
        encoder = self.hf_tokenizer(prompt, return_tensors="pt", truncation=True)

        output = self.hf_model.generate(
            **encoder,
            max_length=48,
            do_sample=True,
            top_p=0.99,
            top_k=50,
            temperature= 1.0,
            num_return_sequences=5,
            repetition_penalty=1.2,
        )

        resultant = self.hf_tokenizer.batch_decode(output, skip_special_tokens=True)
        
        seen = set()
        unique = []
        translator = str.maketrans('', '', string.punctuation)

        for list_sentence in resultant:
            list_sentence = list_sentence.lower().strip()
            list_sentence = list_sentence.translate(translator)
            while (list_sentence[-1:] == ' '):
                list_sentence = list_sentence[:-1]
            seen.add(list_sentence)

        for set_sentence in seen:
            unique.append(set_sentence)

        return unique


    def infer_rating_meaning(self) -> str:
        """Translates numerical rating scores into descriptive, paraphrased sentiment.

        Calculates the aggregate review score and maps it to a sentiment category 
        (ranging from 'Very Negative' to 'Very Positive'). To avoid repetitive 
        output, the final description is passed through an AI paraphrasing 
        engine and a random variation is selected.

        Returns:
            str: A randomly selected paraphrased sentence describing the 
                overall service sentiment.
        """
        overall_rating = self.__calculate_all_review()

        def generate_sentence(rating_summ):
            return f"For {self.__service_name}: " + random.choice(self.__paraphrase_statement(rating_summ)).strip()

        if 1.0 <= overall_rating < 2.0:
            return generate_sentence("Overall sentiment is very negative, indicating widespread dissatisfaction among users.")
        elif 2.0 <= overall_rating < 3.0:
            return generate_sentence("Overall sentiment is negative, suggesting notable dissatisfaction across reviews.")
        elif 3.0 <= overall_rating < 4.0:
            return generate_sentence("Overall sentiment is mixed, reflecting a balance of positive and negative feedback.")
        elif 4.0 <= overall_rating < 5.0:
            return generate_sentence("Overall sentiment is positive, indicating general user satisfaction.")
        else:
            return generate_sentence("Overall sentiment is very positive, reflecting strong user approval and satisfaction.")