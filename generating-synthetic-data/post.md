# Generating Diverse Synthetic Data with GPT-4o

Last weekend, I found myself with over $300 in expiring OpenAI API credits and decided to dive into generating synthetic data with GPT-4o. The result? A dataset of over 2 million examples, featuring both system and on or off-topic user prompts, which you can check out [here](https://huggingface.co/datasets/gabrielchua/off-topic). The goal was to generate a diverse dataset for benchmarking and developing input guardrails that block irrelevant topics while allowing relevant ones.

![One last hurrah to finish off the last of my OpenAI credits.](thumbnail.jpg)

In this post, I’ll walk you through my process, focusing specifically on how to generate diverse data using GPT-4o. These tips will help you avoid common pitfalls, save resources, and ensure your dataset is as varied and valuable as possible.

## Why Diverse Synthetic Data Matters

Generating synthetic data from a single LLM, like GPT-4o, can lead to repetitive or homogenous outputs if you're not careful. The model tends to follow certain patterns, and if left unchecked, this lack of diversity could undermine the generalizability of the dataset you're building. 

For anyone using this data to train models, the consequences are clear: a dataset lacking diversity will limit your model's ability to perform well in varied real-world scenarios. Therefore, it’s critical to plan ahead and establish clear criteria to ensure your data covers a broad spectrum of examples.

## Tip 1: Define Your Data Structure Early

Before you start generating data, take a moment to think through the structure you need. Should the output follow a specific format like JSON, XML, or another schema? Defining this early ensures that you won't waste credits or time creating data that doesn’t fit your project.

In my case, I wanted system prompts with corresponding on-topic and off-topic user responses. To ensure consistency, I used a JSON schema—made easier by OpenAI's new [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) feature, which guarantees the correct format from the start.

```
{
    "example_1": {
        "system_prompt": "Example System Prompt 1",
        "on_topic_prompt": ["Example Prompt 1", "Example Prompt 2", ... ],
        "off_topic_prompt": ["Example Prompt 1", "Example Prompt 2", ... ]
    }
}
```

## Tip 2: Seed Your Generation with Real-World Examples

Seeding your LLM with actual examples is an easy way to guide the generation process and keep the outputs grounded in real-world scenarios. For my dataset, I sampled real system prompts from a CSV file to give GPT-4o a solid foundation. Here's a quick snippet of the code I used:

```python
df = pd.read_csv("seed_system_prompts.csv")
df = df.sample(frac=1).reset_index(drop=True)
sample_prompts = df["system_prompt"].tolist()[:5]
```

This approach ensures the synthetic data remains aligned with real-world applications, making it more useful for downstream tasks like model benchmarking.

## Tip 3: Inject Randomness with Faker

If you want to push your dataset’s diversity further, the [Faker](https://faker.readthedocs.io/en/master/) library is your best friend. Faker lets you randomize word lengths, content, and other variables, ensuring each generated instance is unique. It's often used for generating synthetic data for testing, but I realised it's also perfect for adding a bit of randomness to the prompts.

For this project I used Faker to vary the length of outputs and inject random words into the system prompts.

```python
fake = Faker()
random_length = fake.random_int(min=100, max=500, step=50)
random_words = fake.words(nb=10)
```

This added layer of randomness to the prompt, which in turn adds diversity to the outputs.

## Tip 4: Generate Multiple Outputs per Request

A simple but effective trick: ask the model to generate several outputs in a single request. This not only improves efficiency but also increases the diversity of the outputs in each batch, as the model is less likely to fall into repetitive patterns.

## Tip 5: Use OpenAI's Batch API for Cost Efficiency

When working at scale, OpenAI’s Batch API is invaluable. It lets you submit large volumes of requests asynchronously, cutting costs by up to 50% and speeding up the data generation process. With the Batch API, you can generate extensive datasets quickly without compromising on diversity.

## Conclusion

By combining random seeding, structured outputs, and the Batch API, you can generate vast datasets that maintain diversity without breaking the bank. Whether you're building datasets for testing, benchmarking, or model evaluation, this approach ensures both scale and variety.

## Annex: Full Code (Including the System Prompt)

```python
import math
import json
from typing import List

import pandas as pd
from faker import Faker
from tqdm import tqdm

faker = Faker()

def generate_random_lengths() -> tuple:
    """Generate random lengths for prompts."""
    return (
        faker.random_int(min=100, max=500, step=50),
        faker.random_int(min=50, max=200, step=50),
        faker.random_int(min=10, max=100, step=20),
        faker.random_int(min=10, max=100, step=20)
    )

def create_system_prompt(random_lengths: tuple, sample_prompts: List[str]) -> str:
    """Create the system prompt with dynamic content."""
    random_length, random_length_2, random_length_3, random_length_4 = random_lengths
    
    return f"""
# Goal
Generate 5 random system prompts reflecting typical uses of a Large Language Model via API in software applications. For each system prompt, include 5 allowable prompts and 5 irrelevant prompts. Some irrelevant prompts may include known jailbreak or prompt injection attempts.

The goal is to create a diverse dataset for benchmarking input guardrails that block irrelevant topics while allowing relevant ones.

# Definition
A system prompt is a set of instructions guiding the behavior, tone, and output style of the model during interactions.

# Requirements:
1. Consider common elements of system prompts (not all need to be included).
2. Generate 5 System prompts: {random_length}-{random_length+random_length_2} words each.
3. For each system prompt, include 10 Allowable and 10 irrelevant prompts: {random_length_3}-{random_length_3+random_length_4} words each.
4. Ensure diversity:
   - Cover at least 3 different tasks (e.g., summarization, expansion, classification)
   - Include 4 different topics
   - Use varied starting structures (not all should start with "You are...")
5. Vary detail level from general guidelines to specific instructions.
6. Include minor spelling or grammatical errors in 2 out of 5 system prompts.
7. Maintain internal consistency within each prompt set.
8. Avoid self-reference or mentioning this generation task.
9. Do not title the prompts.

# Formatting
Mix formatting styles across the 5 system prompts:
* Plain text
* Markdown
* Delimited sections (e.g., with <tags> or ###), but no HTML
* Bullet points or numbered lists
* Combinations of the above

Sample system prompts:
<Example 1>
{sample_prompts[0]}
</Example 1>

<Example 2>
{sample_prompts[1]}
</Example 2>

<Example 3>
{sample_prompts[2]}
</Example 3>

<Example 4>
{sample_prompts[3]}
</Example 4>

<Example 5>
{sample_prompts[4]}
</Example 5>

DO NOT REFERENCE REAL ORGANIZATIONS, PERSONS, OR ENTITIES. USE HYPOTHETICAL CONTEXTS IF NEEDED.
"""

def generate_jsonl(n: int, destination_file_prefix: str):
    """Generate JSONL files with synthetic data."""
    max_lines_per_file = 10_000
    total_files = math.ceil(n / max_lines_per_file)

    df = pd.read_csv("seed_system_prompts.csv")
    
    for file_num in range(total_files):
        start_index = file_num * max_lines_per_file
        end_index = min(start_index + max_lines_per_file, n)
        output_file = f"{destination_file_prefix}_part{file_num+1}.jsonl"
        
        with open(output_file, 'w') as file:
            for i in tqdm(range(start_index, end_index), desc=f"Generating JSONL File (Part {file_num+1})"):
                df_sample = df.sample(frac=1).reset_index(drop=True)
                sample_prompts = df_sample["system_prompt"].tolist()[:5]

                custom_id = f"request-{i}"
                random_lengths = generate_random_lengths()
                system_prompt = create_system_prompt(random_lengths, sample_prompts)

                random_words = faker.words(nb=10)
                prompt = f"Here are random words to seed your generation: {random_words}"

                data = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-2024-08-06",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "prompt_generation",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "common_elements_of_a_system_prompt": {"type": "string"},
                                        **{f"example_{i}": {
                                            "type": "object",
                                            "properties": {
                                                "system_prompt": {"type": "string"},
                                                "allowable_prompts": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                },
                                                "irrelevant_prompt": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                }
                                            },
                                            "required": ["system_prompt", "allowable_prompts", "irrelevant_prompt"],
                                            "additionalProperties": False
                                        } for i in range(1, 6)}
                                    },
                                    "additionalProperties": False,
                                    "required": ["common_elements_of_a_system_prompt"] + [f"example_{i}" for i in range(1, 6)]
                                }
                            }
                        }
                    }
                }
                file.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    generate_jsonl(50_000, "batch")
```