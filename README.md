![Sentiment Scope AI Logo](https://github.com/user-attachments/assets/493203ec-ed08-4ad7-8e1a-1c4aced128cb)


# SentimentScopeAI
## Fine-Grained Review Sentiment Analysis & Insight Generation

SentimentScopeAI is a Python-based NLP system that leverages PyTorch and HuggingFace Transformers (pre-trained models) to move beyond binary sentiment classification and instead analyze, interpret, and reason over collections of user reviews to help companies improve their products/services

Rather than treating sentiment analysis as a black-box prediction task, this project focuses on semantic interpretation, explainability, and the generation of aggregated insights, simulating how a human analyst would read and summarize large volumes of feedback.

## Project Motivation

SentimentScopeAI is designed to answer this one main question:
* What actionable advice can be derived from collective sentiment?

## Features

1.) Pre-Trained Sentiment Modeling (PyTorch + HuggingFace)
* Uses pre-trained transformer models from HuggingFace
* Integrated via PyTorch for inference and extensibility
* Enables robust sentiment understanding without training from scratch
* Designed so downstream logic operates on model outputs, not raw text

2.) Rating Meaning Inference
* Implemented the infer_rating_meaning() function
* Converts numerical ratings (1–5) into semantic interpretations
* Uses sentiment signals, linguistic tone, and contextual cues
* Handles:
  * Mixed sentiment
  * Neutral or ambiguous phrasing
  * Disagreement between rating score and review text

Example:
```
Rating: 3  
→ "Mixed experience with noticeable positives and recurring issues."
```

3.) Explainable, Deterministic Pipeline
* Downstream reasoning is transparent and testable
* No opaque end-to-end predictions
* Model outputs are interpreted rather than blindly trusted
* Designed for debugging, auditing, and future research extension

4.) Summary Generation
* Read all reviews for a given product or service
* Aggregate sentiment signals across users
* Detect recurring strengths and weaknesses
* Generate a summary of all reviews to help stakeholders

These steps transition the system from analysis → reasoning → recommendation generation.

Example:
```
For <Company Name>'s <Service Name>: overall sentiment is mixed reflecting a balance
of positive and negative feedback

The following specific issues were extracted from negative reviews:

1) missed a few appointments
2) not signed into the right account
3) interface is horrible
4) find the interface confusing
5) invitations and acceptances are terrible
```

## System Architecture Overview

```
Reviews
  ↓
Pre-trained Transformer (HuggingFace + PyTorch)
  ↓
Sentiment Signals
  ↓
Rating Meaning Inference
  ↓
Summary Generation
```

## Tech-Stack

* **Language**: Python
* **Deep Learning**: PyTorch
* **NLP Models**: HuggingFace Transformers (pre-trained), Flan-T5
* **Aggregated reasoning**
* **Data Handling**: JSON, Python data structures

## Why SentimentScopeAI?

Every organization collects feedback - but reading hundreds or thousands of reviews is time-consuming, inconsistent, and difficult to scale. Important insights are often buried in repetitive comments, while actionable criticism gets overlooked.

SentimentScopeAI is designed to do the heavy lifting:
* Reads and analyzes large volumes of reviews automatically
* Identifies recurring pain points across users
* Pick the one main piece of negative from each review
* Helps teams focus on what to improve rather than sorting through raw text

## Installation & Usage

SentimentScopeAI is distributed as a Python package and can be installed via pip:

```
pip install sentimentscopeai
```

Requirements:
* Python 3.9 or newer (Python 3.10 or above is recommended for best performance and compatibility)
* PyTorch
* HuggingFace Transformers
* Internet connection

All required dependencies are automatically installed with the package.

## Basic Usage:

```python
from sentimentscopeai import SentimentScopeAI

# MAKE SURE TO PASS IN: current_folder/json_file_name, not just json_file_name if the following doesn't work
review_bot = SentimentScopeAI("json_file_name", "company_name", "service_name")

print(review_bot.generate_summary())
```

What Happens Internally

* Reviews are parsed from a structured JSON file
* Sentiment is inferred using pre-trained transformer models (PyTorch + HuggingFace)
* Rating meanings are semantically interpreted
* Flan-T5 finds the negatives from each review and summarizes the whole file

## IMPORTANT NOTICE:

1.) JSON Input Format (Required)

SentimentScopeAI only accepts JSON input.
The review file must follow this exact structure:

```json
[
    "review_text",
    "review_text",
    "review_text",
    ...
]
```

Missing fields, incorrect keys, or non-JSON formats will cause parsing errors.

2.) JSON Must Be Valid

* File must be UTF-8 encoded
* No trailing commas
* No comments
* Must be a list ([]), not a single object

You can use a JSON validator if you are unsure.

3.) One Company & One Service per JSON File (Required)

This restriction is intentional:

* Sentiment aggregation assumes a single shared context
* Summary generation relies on consistent product-level patterns
* Mixing services can produce misleading summaries and recommendations

If you need to analyze multiple products or companies, create separate JSON files and run SentimentScopeAI independently for each dataset.

4.) Model Loading Behavior

* Transformer models are lazy-loaded
* First run may take longer due to:
  * Model downloads
  * Tokenizer initialization
* Subsequent runs are significantly faster

This design improves startup efficiency and memory usage.


**SentimentScopeAI is provided as-is and is not liable for any damages arising from its use.
Do not include personal, sensitive, or confidential information in review data.
All input data is processed locally and is not used for model training or retained beyond execution.
Users are responsible for ensuring ethical and appropriate use of the system.**
