import torch
import json
import os
import string
import random
import textwrap
import time
import sys
import threading
from difflib import SequenceMatcher
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
    __device = None
    __notable_negatives = []
    __extraction_model = None
    __extraction_tokenizer = None
    __company_name = None
    __service_name = None
    __stop_timer = None
    __timer_thread = None

    def __init__(self, file_path, company_name, service_name):
        """
            Initialize the SentimentScopeAI class with the specified JSON file path, company's name, and service's name.

            Args:
                - file_path (str): specified JSON file path
                - company_name (str): name of the company being reviewed
                - service_name (str): name of the company's service/product being reviewed

            Returns:
                tuple: A tuple containing the total number of reviews and the average star rating.
        """
        self.__hf_model_name = "Vamsi/T5_Paraphrase_Paws"
        self.__pytorch_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.__extraction_model_name = "google/flan-t5-large"
        self.__company_name = company_name
        self.__service_name = service_name
        self.__json_file_path = os.path.abspath(file_path)
        print("""
        ─────────────────────────────────────────────────────────────────────────────
        SentimentScopeAI can make mistakes. This AI may produce incomplete summaries,
        misclassify sentiment, or categorize positive feedback as negative. Please
        verify critical insights before making decisions based on this analysis.
        ─────────────────────────────────────────────────────────────────────────────
        """)
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__stop_timer = threading.Event()
        self.__timer_thread = threading.Thread(target=self.__time_threading)
        self.__timer_thread.start()

    @property
    def hf_model(self):
        """Lazy loader for the Paraphrase Model."""
        if self.__hf_model is None:
            self.__hf_model = AutoModelForSeq2SeqLM.from_pretrained(self.__hf_model_name)
        return self.__hf_model

    @property
    def hf_tokenizer(self):
        """Lazy loader for the Paraphrase Tokenizer."""
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
   
    def __time_threading(self) -> None:
        """Time Threading for elapsed timer while SentimentScopeAI processes"""
        start_time = time.time()
        while not self.__stop_timer.is_set():
            elapsed_time = time.time() - start_time
            mins, secs = divmod(elapsed_time, 60)
            hours, mins = divmod(mins, 60)
           
            timer_display = f"SentimentScopeAI is processing (elapsed time): {int(hours):02}:{int(mins):02}:{int(secs):02}"
            sys.stdout.write('\r' + timer_display)
            sys.stdout.flush()
           
            time.sleep(0.1)

    def __get_predictive_star(self, text: str) -> int:
        """
            Predict the sentiment star rating for the given text review.

            Args:
                text (str): The text review to analyze.
            Returns:
                int: The predicted star rating (1 to 5).
        """
        max_len = getattr(self.pytorch_tokenizer, "model_max_length", 512)

        inputs = self.pytorch_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(self.__device)

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
                single_review_rating = self.__get_predictive_star(entry)
                sum += single_review_rating
                num_reviews = i
        return (sum / num_reviews) if num_reviews != 0 else 0
   
    def __paraphrase_statement(self, statement: str) -> list[str]:
        """
            Generates multiple unique paraphrased variations of a given string.

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
        """
            Translates numerical rating scores into descriptive, paraphrased sentiment.

            Calculates the aggregate review score and maps it to a sentiment category
            (ranging from 'Very Negative' to 'Very Positive'). To avoid repetitive
            output, the final description is passed through an AI paraphrasing
            engine and a random variation is selected.

            Args:
                None

            Returns:
                str: A randomly selected paraphrased sentence describing the
                    overall service sentiment.
        """
        overall_rating = self.__calculate_all_review()

        if overall_rating is None:
            return "JSON FILE PATH IS UNIDENTIFIABLE, please try inputting the name properly (e.g. \"companyreview.json\")."

        def generate_sentence(rating_summ):
            return f"For {self.__company_name}'s {self.__service_name}: " + random.choice(self.__paraphrase_statement(rating_summ)).strip()

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

    def __delete_duplicate(self, issues: list[str]) -> list[str]:
        """
            Filters out duplicate and near-duplicate issue strings using fuzzy matching.

            The method normalizes strings by converting them to lowercase and stripping
            whitespace. It ignores issues that are empty or contain two or fewer words.
            A string is considered a duplicate if its similarity ratio (via SequenceMatcher)
            is greater than 0.75 compared to any already accepted issue.

            Args:
                issues (list[str]): A list of raw issue descriptions to be processed.

            Returns:
                list[str]: A list of unique, normalized issue strings that met the similarity requirements.
        """
        if not issues:
            return []

        result = []
        for issue in issues:
            if not issue:
                continue
            issue = issue.lower().strip()
           
            is_dup = any(SequenceMatcher(None, issue, existing).ratio() >= 0.40 for existing in result)

            if not is_dup:
                result.append(issue)
        return result
   
    def __validate_issue(self, extracted_issue: str) -> bool:
        """
            Determine whether an extracted line represents a true negative issue.

            This method acts as a polarity gate after issue extraction, filtering out
            positives, neutral statements, feature descriptions, and vague suggestions
            that were incorrectly labeled as issues.

            Args:
                extracted_issue (str): A single line extracted as a potential issue.

            Returns:
                bool: True if the line is a clear negative issue, False otherwise.
        """

        if not extracted_issue:
            return False
       
        vprompt = f"""
        You are a strict polarity verifier for extracted "issues" across many industries.

        Task:
        Given ONE extracted line, decide if it is truly a NEGATIVE complaint/problem.

        Return EXACTLY one token: YES or NO

        Rules:
        - Output YES only if the line explicitly states a problem, failure, drawback, frustration, harm, or limitation.
        - Output NO for praise, neutral facts, feature descriptions, or wishes/suggestions without a stated problem.
        - Mixed lines: output NO only if the negative part is explicit.
        - No inference. If ambiguous, output NO.

        Few-shot examples:

        1) INPUT: "The dashboard times out and loses my changes."
        OUTPUT: YES

        2) INPUT: "Package arrived late and tracking never updated."
        OUTPUT: YES

        3) INPUT: "I got charged an unexpected fee and support couldn't explain it."
        OUTPUT: YES

        4) INPUT: "Flight was canceled with little notice and rebooking took hours."
        OUTPUT: YES

        5) INPUT: "Internet drops daily and speeds are far below what I pay for."
        OUTPUT: YES

        6) INPUT: "Appointment started 45 minutes late and I couldn't reach anyone."
        OUTPUT: YES

        7) INPUT: "Delivery was fast and the order was correct."
        OUTPUT: NO

        8) INPUT: "Graphics are amazing and performance is smooth."
        OUTPUT: NO

        9) INPUT: "Setup was easy and it integrates well with Alexa."
        OUTPUT: NO

        10) INPUT: "Content is well-structured and easy to follow."
            OUTPUT: NO

        11) INPUT: "Timesheets are easy to submit and approvals are quick."
            OUTPUT: NO

        12) INPUT: "Documentation is clear and examples are helpful."
            OUTPUT: NO

        Now classify:

        INPUT: "{extracted_issue}"

        OUTPUT:
        """.strip()

        validator_in = self.extraction_tokenizer(vprompt, return_tensors="pt", max_length=512, truncation=True).to(self.__device)
        validator_out = self.extraction_model.generate(**validator_in, max_new_tokens=5, num_beams=1, do_sample=False)
        verdict = self.extraction_tokenizer.decode(validator_out[0], skip_special_tokens=True).strip().upper()
        return verdict == "YES"
   
    def __extract_negative_aspects(self, review: str) -> list[str]:
        """
            Extract actionable negative aspects from a review using AI-based text generation.
           
            This method uses the Flan-T5 language model to identify specific, constructive
            problems mentioned in a review. Unlike simple sentiment analysis, this extracts
            concrete issues that describe what is broken, missing, or difficult - filtering
            out vague emotional words like "horrible" or "bad".
           
            Args:
                review (str): The review text to analyze for negative aspects.
           
            Returns:
                list[str]: A list of specific problem phrases extracted from the review.
           
            Note:
                This method uses the Flan-T5 model which is loaded lazily on first use.
                Processing time depends on review length and available hardware (CPU/GPU).
                Very short outputs (<=3 characters) are filtered out as likely artifacts.
        """
        if not review or review.isspace():
            return []

        prompt = f"""
        Task: Extract ONE specific operational issue from the review in 6-14 words.

        Rules:
        - if there is no clear issue, only vague emotions, or positive review, then Output: none
        - Output the concrete problem using ONLY words from the review, but be concise
        - Include specific details (numbers, times, items) when mentioned
        - Keep role descriptions, if there are any, BUT remove person names

        Examples:

        Review: "Waited 2 hours past scheduled time with no explanation given."
        Answer: waited 2 hours past scheduled time no explanation

        Review: "Terrible experience, worst place ever, never again!"
        Answer: none

        Review: "Was charged $50 extra fee that wasn't mentioned upfront."
        Answer: charged 50 dollar extra fee not mentioned upfront

        Review: "Staff was extremely rude and unprofessional throughout."
        Answer: none

        Review: "Ordered item A but received item B, return process unclear."
        Answer: ordered item a received item b return unclear

        Review: "System crashed three times during checkout process."
        Answer: system crashed three times during checkout

        Review: "Amazing service, highly recommend to everyone!"
        Answer: none

        Review: "Called customer support 5 times, never got callback as promised."
        Answer: called support 5 times never got promised callback

        Review: "Product arrived damaged with missing parts, no replacement offered."
        Answer: product arrived damaged missing parts no replacement offered

        Review: "Unbelievable how bad this was, absolutely horrible."
        Answer: none

        Review: "{review}"
        Answer:
        """.strip()

        inputs = self.extraction_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.__device)


        outputs = self.extraction_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        result = self.extraction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if result.strip().lower() in ['none', 'none.', 'no problems', '']:
            return []
       
        issues = []
        for line in result.split('\n'):
            line = line.strip()
            line = line.lstrip('•-*1234567890.) ')
            if line and len(line) > 3:
                issues.append(line)
       
        if not issues:
            return []


        if not (self.__validate_issue(issues[0])):
            return []
       
        return issues

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

            Args:
                None

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
                    for part in self.__extract_negative_aspects(entry):
                        self.__notable_negatives.append(part)
                    reviews.append(entry)
        except FileNotFoundError:
            return ("JSON file path is unidentifiable, please try inputting the name properly (e.g. \"companyreview.json\").")
        except json.JSONDecodeError:
            return ("Could not decode JSON file. Check for valid JSON syntax (look at GitHub/PyPi Readme Instructions).")
        except PermissionError:
            return ("Permission denied to open the JSON file.")
        except Exception as e:
            return (f"An unexpected error occured: {e}")
       
        self.__notable_negatives = self.__delete_duplicate(self.__notable_negatives)

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
       
        self.__stop_timer.set()
        self.__timer_thread.join()
        print()
        print()

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