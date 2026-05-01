![Sentiment Scope AI Logo](https://github.com/user-attachments/assets/493203ec-ed08-4ad7-8e1a-1c4aced128cb)


# SentimentScopeAI
## Fine-Grained Review Sentiment Analysis & Insight Generation

SentimentScopeAI is a Python-based NLP system that leverages PyTorch and HuggingFace Transformers (pre-trained models) to analyze, interpret, and point out concerns from customer reviews to help companies improve their product/services.

## Project Motivation

SentimentScopeAI is designed to answer this one main question:
* What concerns can be derived from a massive set of collective sentiment?

Example:
```
For <Company Name>'s <Service Name>: overall sentiment is mixed reflecting a balance
of positive and negative feedback

The following specific issues were extracted from negative reviews:

MOST FREQUENT CONCERNS:
1) missed a few appointments (5 customers)
2) not signed into the right account (3 customers)
3) interface is horrible (2 customers)

OTHER CONCERNS:
4) find the interface confusing
5) invitations and acceptances are terrible
```

## Tech-Stack
* **Language**: Python
* **Deep Learning**: PyTorch
* **NLP Models**: HuggingFace Transformers (pre-trained), Flan-T5
* **Web Scraping**: Playwright, Playwright-Stealth, SeleniumBase
* **Aggregated Reasoning**: Multi-model ensemble approach
* **Data Handling**: JSON, Python data structures

## Why SentimentScopeAI?

Every organization collects feedback - but reading hundreds or thousands of reviews is time-consuming, inconsistent, and difficult to scale. Important insights are often buried in repetitive comments, while actionable criticism gets overlooked.

SentimentScopeAI is designed to do the heavy lifting:
* Reads and analyzes large volumes of reviews automatically
* Identifies recurring pain points across users
* Pick the one main piece of concern from each review (if there are any)
* Helps teams focus on what to improve rather than sorting through raw text

## Installation & Usage

SentimentScopeAI is distributed as a Python package and can be installed via pip:

```
pip install sentimentscopeai
```

Requirements:
* Python 3.12.0 or higher (IMPORTANT)
* Internet connection

All required dependencies are automatically installed with the package.

## Basic Usage:

```python
from sentimentscopeai import SentimentScopeAI

# MAKE SURE TO PASS IN: <current_folder/file_name.json>, not just <file_name.json> if the following doesn't work
review_bot = SentimentScopeAI("file_name.json", "company_name", "service_name")

print(review_bot.generate_summary(batch_size=8)) # read the rest of the README to learn more about batch_size
```

What Happens Internally

* Reviews are parsed from a structured JSON file
* Sentiment is inferred using pre-trained transformer models (PyTorch + HuggingFace)
* Rating meanings are semantically interpreted
* Flan-T5 finds the negatives from each review and summarizes the whole file


## Runtime Analysis


![ssAI Runtime Performance Metric Graph](https://github.com/user-attachments/assets/0f43f3da-92d7-4356-b80a-0a095a91b4c6)

**NOTE**
- **Time Complexity:** Θ(N) with empirical R² = 0.998
- **Throughput:** ~100 reviews/minute
- **Benchmark Hardware:** NVIDIA RTX 3050 (CUDA-enabled)
- **Performance Variance:** Execution time scales with GPU capabilities, VRAM, and CUDA compatibility


## Important Notice:

1.) Batch Size

A newly added feature that allows users to configure the batch size, which determines the number of reviews processed simultaneously by the AI models before updating internal computations. The optimal batch size varies depending on hardware specifications. For lower-end dedicated GPUs or integrated GPUs with limited VRAM or system RAM, it is recommended to use a lower batch_size value (1, 2, or a maximum of 4) when calling the generate_summary() method. Conversely, if you are using a more capable GPU or computer with higher specifications, a higher batch_size value (8, 16, or 32) will yield significantly better performance. 

However, it is NOT recommended to specify batch_size values exceeding 64, as this may cause system instability or application crashes. 

The ability to configure batch_size is intentionally exposed to users to ensure maximum control over the software's behavior, allowing optimization based on individual hardware capabilities and performance requirements.

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
Check out: [https://jsonlint.com/]

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

## Web Scraping Feature

SentimentScopeAI now includes an **optional automated review import feature** that can scrape reviews directly from Yelp for analysis.

### Additional Setup for Web Scraping

If you want to use the automated scraping feature, install the required browser:
```bash
playwright install chromium
```

### Example Usage (after playwright install chromium)
```python
from sentimentscopeai import SentimentScopeAI

# MAKE SURE TO PASS IN: <current_folder/file_name.json>, not just <file_name.json> if the following doesn't work
review_bot = SentimentScopeAI("file_name.json", "company_name", "service_name")

review_bot.import_yelp_reviews("https://www.yelp.com/biz/business-name-here#reviews")

print(bot.generate_summary())
```

### Supported Platforms
- Yelp Reviews [https://www.yelp.com/]

### IMPORTANT NOTES
- Scraping may take several minutes to an hour for businesses with many reviews
- The feature includes anti-detection measures and random delays
- Reviews are automatically cleaned and formatted
- For best results, ensure a stable internet connection
- For faster processing of a large dataset, a dedicated NVIDIA GPU with CUDA is highly recommended

### Disclaimer: 

SentimentScopeAI is provided **as-is** and is **not liable** for any damages arising from its use. All input data is **processed locally** and is **not used for model training** or retained beyond execution. **Do not include personal, sensitive, or confidential information** in review data. SentimentScopeAI **may produce incomplete summaries or misclassify sentiment**. Always **verify critical insights** before making business decisions. 

**Web Scraping Notice:** SentimentScopeAI is **not affiliated with, endorsed by, or partnered with Yelp Inc.** Users are **solely responsible for complying with Yelp's Terms of Service** and applicable laws. This feature is provided for **research and personal use only**. Users are **responsible for ensuring ethical and appropriate use** of this system.
