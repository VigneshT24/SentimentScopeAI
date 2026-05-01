import torch
import json
import os
import string
import random
import textwrap
import time
import sys
import threading
import warnings
import logging

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from seleniumbase import sb_cdp
from difflib import SequenceMatcher
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM,
                          T5ForConditionalGeneration, T5Tokenizer, set_seed)


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

        Web scraping feature: SentimentScopeAI is not affiliated with, endorsed by,
        or partnered with Yelp Inc. Users are responsible for complying with Yelp's
        Terms of Service. This feature is provided for research and personal use only.
        ─────────────────────────────────────────────────────────────────────────────
        """)
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__stop_timer = threading.Event()
        self.__timer_thread = threading.Thread(target=self.__time_threading)

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

    def __calculate_all_review(self) -> int:
        """
            Calculate and print the predicted star ratings for all reviews in the JSON file.

            Args:
                None
            Returns:
                tuple: A tuple containing the total number of reviews and the average star rating.
        """
        # don't need try-catch because it is handled in generate_summary()
        with open(self.__json_file_path, 'r', encoding="utf-8") as reviews_file:
            all_reviews = json.load(reviews_file)

            if not all_reviews:
                return 0

            total_stars = 0
            batch_size = 32

            for i in range(0, len(all_reviews), batch_size):
                batch = all_reviews[i : (i + batch_size)]

                inputs = self.pytorch_tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.__device)

                with torch.no_grad():
                    outputs = self.pytorch_model(**inputs)

                predictions = torch.argmax(outputs.logits, dim=-1)
                total_stars += (predictions + 1).sum().item()

            return total_stars / len(all_reviews)

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
        set_seed(random.randint(0, 2 ** 32 - 1))

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
            temperature=1.0,
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

    def __extract_negative_aspects_batch(self, batch_of_reviews: list) -> list[str]:
        """
        Extract actionable negative aspects from multiple reviews simultaneously using batch processing.

        This method processes multiple reviews in parallel on the GPU, significantly improving
        throughput compared to sequential processing.

        Args:
            batch_of_reviews (list[str]): List of review texts to analyze for negative aspects

        Returns:
            list[str]: All extracted problem phrases from all reviews in the batch
        """

        if not batch_of_reviews:
            return []

        prompts = []
        valid_indices = []

        for index, review in enumerate(batch_of_reviews):
            if not review or review.isspace():
                continue

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

            prompts.append(prompt)
            valid_indices.append(index)

        if not prompts:
            return []

        inputs = self.extraction_tokenizer(
            prompts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.__device)

        outputs = self.extraction_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        all_issues = []

        for output in outputs:
            result = self.extraction_tokenizer.decode(output, skip_special_tokens=True)

            if result.strip().lower() in ['none', 'none.', 'no problems', '']:
                continue

            issues = []
            for line in result.split('\n'):
                line = line.strip()
                line = line.lstrip('•-*1234567890.) ')
                if line and len(line) > 3:
                    issues.append(line)

            if not issues:
                continue

            if self.__validate_issue(issues[0]):
                all_issues.extend(issues)

        return all_issues

    def __infer_rating_meaning(self) -> str:
        """
            Translates numerical rating scores into descriptive, paraphrased sentiment.

            Calculates the aggregate review score and maps it to a sentiment category
            (ranging from 'Very Negative' to 'Very Positive'). To avoid repetitive
            output, the final description is passed through an paraphrasing
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
            return f"For {self.__company_name}'s {self.__service_name}: " + random.choice(
                self.__paraphrase_statement(rating_summ)).strip()

        if 1.0 <= overall_rating < 2.0:
            return generate_sentence(
                "Overall sentiment is very negative, indicating widespread dissatisfaction among users.")
        elif 2.0 <= overall_rating < 3.0:
            return generate_sentence(
                "Overall sentiment is negative, suggesting notable dissatisfaction across reviews.")
        elif 3.0 <= overall_rating < 4.0:
            return generate_sentence(
                "Overall sentiment is mixed, reflecting a balance of positive and negative feedback.")
        elif 4.0 <= overall_rating < 5.0:
            return generate_sentence("Overall sentiment is positive, indicating general user satisfaction.")
        else:
            return generate_sentence(
                "Overall sentiment is very positive, reflecting strong user approval and satisfaction.")

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
        - Mixed lines: output NO only if the negative part isn't explicit.
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

        validator_in = self.extraction_tokenizer(vprompt, return_tensors="pt", max_length=512, truncation=True).to(
            self.__device)
        validator_out = self.extraction_model.generate(**validator_in, max_new_tokens=5, num_beams=1, do_sample=False)
        verdict = self.extraction_tokenizer.decode(validator_out[0], skip_special_tokens=True).strip().upper()
        return verdict == "YES"

    def __frequency_rank(self, list_of_concerns) -> list[tuple]:
        freq_groups = []

        for concern in list_of_concerns:
            best_match = -1
            best_ratio = 0

            for i, (freq_phrase, j) in enumerate(freq_groups):
                ratio = SequenceMatcher(None, concern, freq_phrase).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = i

            if best_ratio >= 0.50:
                freq_phrase, count = freq_groups[best_match]
                freq_groups[best_match] = (freq_phrase, count + 1)
            else:
                freq_groups.append((concern, 1))

        return sorted(freq_groups, key=lambda x: x[1], reverse=True)

    def import_yelp_reviews(self, url) -> None:
        """
        Automatically imports customer reviews from a Yelp business page using web scraping.

        This method navigates through all available review pages on Yelp, extracts review text content,
        cleans and formats the data, and saves it to a JSON file. The scraper handles pagination
        automatically and continues until all reviews are retrieved from the business listing.

        Args:
            url (str): The complete Yelp business URL including the reviews section.

        Returns:
            None

        Raises:
            TimeoutError: If the page fails to load or reviews cannot be found within the timeout period.
            IOError: If the JSON file cannot be written due to permissions or disk space issues.
            Exception: If scraping fails due to connectivity issues or changes in Yelp's page structure.

        Note:
            - This feature requires an active internet connection
            - Scraping may take several minutes for businesses with many reviews
            - Reviews are automatically cleaned (newlines removed, whitespace normalized)
            - Be mindful of Yelp's terms of service when using this feature
        """
        if (os.stat(self.__json_file_path).st_size != 0):
            print(f"The file: \"{self.__json_file_path}\" must be empty for 'import_yelp_reviews' to work.")
            sys.exit(1)

        reviews = []

        # set up preprocessing for playwright and seleniumbase
        sb = sb_cdp.Chrome(locale="en")
        endpoint_url = sb.get_endpoint_url()
        json_file = self.__json_file_path
        web_url = url

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            context = browser.contexts[0]
            page = context.pages[0]

            stealth = Stealth()
            stealth.use_sync(context)

            page.goto(web_url)
            time.sleep(random.uniform(2, 4))

            # find the reivew_text's unique identifier for the bot to scrape
            review_selector = "span[class*='raw__'][lang='en']"
            page.wait_for_selector(review_selector, timeout=10000)

            # scrape all the reviews by scraping -> next page -> scraping...
            while True:
                review_texts = page.query_selector_all(review_selector)

                for text in review_texts:
                    text = text.inner_text()
                    cleaned_text = text.replace('\n', ' ').strip()
                    cleaned_text = ' '.join(cleaned_text.split())
                    reviews.append(cleaned_text)

                next_btn = page.query_selector("a.next-link[aria-label='Next']")

                if not next_btn:
                    break

                next_btn.hover()
                time.sleep(random.uniform(1, 2))
                next_btn.click()
                time.sleep(random.uniform(4, 7))

            # safely close the browser once all is done
            browser.close()

        try:
            with open(json_file, "w", encoding="utf-8") as rev_file:
                json.dump(reviews, rev_file, indent=2, ensure_ascii=False)
            print(f"Saved {len(reviews)} reviews to the file \"{json_file}\"")
        except IOError as e:
            print(f"Error saving file: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def generate_summary(self, batch_size) -> str:
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
                batch_size (int): number that indicates how much customer reviews should be
                in a batch per process (recommend 2 or 4 for reducing overhead)

            Returns:
                str
                    A multi-paragraph, text-wrapped sentiment summary suitable for
                    console output, logs, or reports.

            Raises:
                None
                    All exceptions are handled internally with descriptive error
                    messages to prevent interruption of execution.
        """
        self.__timer_thread.start()
        try:
            reviews = []
            with open(self.__json_file_path, 'r', encoding="utf-8") as file:
                company_reviews = json.load(file)

                for i in range (0, len(company_reviews), batch_size):
                    batch = company_reviews[i : (i + batch_size)]

                    batch_issues = self.__extract_negative_aspects_batch(batch)
                    self.__notable_negatives.extend(batch_issues)

                    reviews.extend(batch)

        except FileNotFoundError:
            return ("JSON file path is unidentifiable, please try inputting the name properly (e.g. \"companyreview.json\").")
        except json.JSONDecodeError:
            return ("Could not decode JSON file. Check for valid JSON syntax (look at GitHub/PyPi Readme Instructions).")
        except PermissionError:
            return ("Permission denied to open the JSON file.")
        except Exception as e:
            return (f"An unexpected error occured: {e}")

        resulting_loc = self.__frequency_rank(self.__notable_negatives)

        def format_numbered_list(items):
            if not items:
                return "None found"

            lines = []
            flag = False
            recurring_counter = 1
            other_counter = 1

            lines.append("MOST FREQUENT CONCERNS:")

            for item in items:
                phrase, count = item

                if not flag and count < 2:
                    lines.append("\nOTHER CONCERNS:")
                    flag = True

                if count >= 2:
                    prefix = f"{recurring_counter}) "
                    recurring_counter += 1
                else:
                    prefix = f"{other_counter}) "
                    other_counter += 1

                wrapper = textwrap.TextWrapper(
                    width=70,
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix)
                )

                label = f"{phrase} ({count} customers)" if count >= 2 else phrase
                lines.append(wrapper.fill(label))

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
            parts.append(format_numbered_list(resulting_loc))

        return "\n\n".join(parts)