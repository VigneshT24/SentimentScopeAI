import torch
import json
import os
import string
import random
import textwrap
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, set_seed)

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
    __notable_negatives = []
    __extraction_model = None
    __extraction_tokenizer = None

    def __init__(self, file_path):
        """Initialize the SentimentScopeAI class with the specified JSON file path."""
        self.__hf_model_name = "Vamsi/T5_Paraphrase_Paws"
        self.__pytorch_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.__extraction_model_name = "google/flan-t5-large"
        self.__json_file_path = os.path.abspath(file_path)
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def hf_model(self):
        """Lazy loader for the Paraphrase Model."""
        if self.__hf_model is None:
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
        """Lazy loader for the PyTorch Tokenizer."""
        if self.__pytorch_tokenizer is None:
            self.__pytorch_tokenizer = AutoTokenizer.from_pretrained(self.__pytorch_model_name)
        return self.__pytorch_tokenizer

    @property
    def pytorch_model(self):
        """Lazy loader for the PyTorch Model."""
        if self.__pytorch_model is None:
            self.__pytorch_model = AutoModelForSequenceClassification.from_pretrained(
                self.__pytorch_model_name
            ).to(self.__device)
        return self.__pytorch_model

    @property
    def extraction_model(self):
        """Lazy loader for the Flan-T5 extraction model."""
        if self.__extraction_model is None:
            self.__extraction_model = T5ForConditionalGeneration.from_pretrained(
                self.__extraction_model_name
            ).to(self.__device)
        return self.__extraction_model

    @property
    def extraction_tokenizer(self):
        """Lazy loader for the Flan-T5 tokenizer."""
        if self.__extraction_tokenizer is None:
            self.__extraction_tokenizer = AutoTokenizer.from_pretrained(
                self.__extraction_model_name
            )
        return self.__extraction_tokenizer

    def __get_predictive_star(self, text) -> int:
        """
            Predict the sentiment star rating for the given text review.

            Args:
                text (str): The text review to analyze.
            Returns:
                int: The predicted star rating (1 to 5).
        """
        inputs = self.pytorch_tokenizer(text, return_tensors="pt").to(self.__device)

        with torch.no_grad():
            outputs = self.pytorch_model(**inputs)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

        num_star = prediction + 1
        return num_star

    def __calculate_all_review(self) -> int:
        """
            Calculate and print the predicted star ratings for all reviews in the JSON file.

            Args:
                None
            Returns:
                tuple: A tuple containing the total number of reviews and the average star rating.
        """
        # don't need try-catch because it is handled in generate_summary()
        with open(self.__json_file_path, 'r') as reviews_file:
            all_reviews = json.load(reviews_file)
            sum = 0
            num_reviews = 0
            for i, entry in enumerate(all_reviews, 1):
                single_review_rating = self.__get_predictive_star(entry['review'])
                sum += single_review_rating
                self.__service_name = entry['service_name']
                num_reviews = i
        return (sum / num_reviews) if num_reviews != 0 else 0
    
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
    
    def __infer_rating_meaning(self) -> str:
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

        if overall_rating is None:
            return "JSON FILE PATH IS UNIDENTIFIABLE, please try inputting the name properly (e.g. \"companyreview.json\")."

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

    def __extract_negative_aspects(self, review: str) -> list[str]:
        """
        Extract actionable negative aspects from a review using AI-based text generation.
        
        This method uses the Flan-T5 language model to identify specific, constructive
        problems mentioned in a review. Unlike simple sentiment analysis, this extracts
        concrete issues that describe what is broken, missing, or difficult - filtering
        out vague emotional words like "horrible" or "bad".
        
        The extraction focuses on actionable feedback that can help improve a product
        or service, such as "notifications arrive at wrong times" rather than just
        "notifications are bad".
        
        Args:
            review (str): The review text to analyze for negative aspects.
        
        Returns:
            list[str]: A list of specific problem phrases extracted from the review.
                    Each phrase describes a concrete issue. Returns an empty list
                    if the review is empty, contains only whitespace, or no 
                    problems are identified.
        
        Note:
            This method uses the Flan-T5 model which is loaded lazily on first use.
            Processing time depends on review length and available hardware (CPU/GPU).
            Very short outputs (≤3 characters) are filtered out as likely artifacts.
        """
        if not review or review.isspace():
            return []
        
        prompt = f"""What problems does this review mention? List each as a brief phrase.
        Each problem should be describing what is wrong, DON'T OUTPUT one word lines like "horrible" or "bad".
        Make sure they are CONSTRUCTIVE CRITICISM THAT CAN HELP SOMEONE IMPROVE

        Review: {review}

        Problems mentioned:"""

        inputs = self.extraction_tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.__device)

        outputs = self.extraction_model.generate(
            **inputs,
            max_length=150,
            num_beams=4,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            early_stopping=True
        )

        result = self.extraction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if result.strip().lower() in ['none', 'none.', 'no problems', '']:
            return[]
        
        issues = []
        for line in result.split('\n'):
            line = line.strip()
            line = line.lstrip('•-*1234567890.) ')
            if line and len(line) > 3:
                issues.append(line)
        
        return issues
    
    def output_all_reviews(self) -> None:
        """
            Output all reviews from the JSON file in a formatted manner.

            Args:
                None
            Returns:
                None
        """
        # don't need try-catch because it is handled in generate_summary()
        with open(self.__json_file_path, 'r') as file:
            company_reviews = json.load(file)
            for i, entry in enumerate(company_reviews, 1):
                print(f"Review #{i}")
                print(f"Company Name: {entry['company_name']}")
                print(f"Service Name: {entry['service_name']}")
                print(f"Review: {textwrap.fill(entry['review'], width=70)}")
                print("\n\n")

    def generate_summary(self) -> str:
        """
        Generate a formatted sentiment summary based on user reviews for a service.

        This method reads a JSON file containing user reviews, infers the overall
        sentiment rating, and produces a structured, human-readable summary.
        The summary includes:
            - A concise explanation of the inferred sentiment rating
            - A numbered list of representative negatives mentioned

        Long-form reviews are wrapped to a fixed line width while preserving
        list structure and readability.

        The method is resilient to common file and parsing errors and will
        emit descriptive messages if the input file cannot be accessed or
        decoded properly.

        Returns:
            str
                A multi-paragraph, text-wrapped sentiment summary suitable for
                console output, logs, or reports.

        Raises:
            None
                All exceptions are handled internally with descriptive error
                messages to prevent interruption of execution.
        """
        try:
            reviews = []
            with open(self.__json_file_path, 'r') as file:
                company_reviews = json.load(file)
                for i, entry in enumerate(company_reviews, 1):
                    if self.__get_predictive_star(entry['review']) <= 2:
                        for part in self.__extract_negative_aspects(entry['review']):
                            self.__notable_negatives.append(part)
                    self.__service_name = entry['service_name']
                    reviews.append(entry['review'])
        except FileNotFoundError:
            return ("JSON FILE PATH IS UNIDENTIFIABLE, please try inputting the name properly (e.g. \"companyreview.json\").")
        except json.JSONDecodeError:
            return ("Could not decode JSON file. Check for valid JSON syntax.")
        except PermissionError:
            return ("Permission denied to open the JSON file.")
        except Exception as e:
            return (f"An unexpected error occured: {e}")

        def format_numbered_list(items):
            if not items:
                return "None found"

            lines = []
            for i, item in enumerate(items, start=1):
                prefix = f"{i}) "
                wrapper = textwrap.TextWrapper(
                    width=70,
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix) + "   "
                )
                lines.append(wrapper.fill(str(item)))
            return "\n".join(lines)

        rating_meaning = self.__infer_rating_meaning()
        
        parts = [textwrap.fill(rating_meaning, width=70)]

        if self.__calculate_all_review() >= 4:
            parts.append(
                textwrap.fill(
                    "Since the overall rating is good, I don't have any notable negatives to mention.",
                    width=70))
        else:
            parts.append(
                textwrap.fill(
                    "The following reviews highlight some concerns users have expressed:",
                    width=70))
            parts.append(format_numbered_list(self.__notable_negatives))

        return "\n\n".join(parts) 