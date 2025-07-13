# Fine-tuning TinyLlama with Axolotl: A Beginner's Guide

I recently signed up for the [LLM Fine-Tuning course/conference](https://maven.com/parlance-labs/fine-tuning). Prior to this, most of my fine-tuning experience was either through OpenAI's fine-tuning API for GPT 3.5, or through the starter scripts in the [MLX Examples repo](https://github.com/ml-explore/mlx-examples).

One tool introduced I learnt in the course was Axolotl. Basically, with Axolotl, I can easily fine-tune LLMs through the CLI based on configurations defined in a `.yml` file. Within the [Axolotl GitHub repo](https://github.com/OpenAccess-AI-Collective/axolotl), there are many helpful example `.yml` files one can easily re-use.

To run Axolotl, I'm using Jarvis Labs - a GPU cloud provider - that was also one of the generous GPU credits sponsors for the course. What I appreciated was how I could easily launch a [template](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://jarvislabs.ai/templates/axolotl&ved=2ahUKEwipzfDusOCGAxV54TgGHYuvAtcQFnoECBAQAQ&usg=AOvVaw3yYsouWYX7jUn1HYOy4_TF) with Jupyter Notebook instance with the Axolot repo already cloned and other dependencies installed.

To get my hands dirty and apply the above, I did up this toy example where I fine-tuned TinyLlama to generate David Attenborough style narration. The final model can be found [here](https://huggingface.co/cyzgab/TinyLlama-1.1B-DavidAttenborough) on HuggingFace, which you can also try. 

The aim of this blog post is to document my learning process doing the above. Overall, the process took about 1 hour to prepare the data and fine-tune the model, though the actual fine-tuning took about 15 minutes. As such, this post also does lean more towards the data generation and preparation steps. Additionally, in terms of cost, the fine-tuning alone took slightly less than 1 USD.

![](thumbnail.jpg)

# Step 1: Generating the synthetic data

We begin by generating the conversation pairs. For this example, I used OpenAI's models, specifically through the [Batch API](https://platform.openai.com/docs/guides/batch). This helps to reduce cost.

For the Batch API, we need to provide a `.jsonl` file where each line is the POST request to OpenAI's chat completion endpoint. Here is a sample of what that looks like:

```bash
{"custom_id": "0", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4-turbo-2024-04-09", "messages": [{"role": "system", "content": "Imagine you are David Attenborough. I will give you an activity, and you will give a ~300 word narration for a documentary."}, {"role": "user", "content": "A young girl planting flowers in her backyard garden."}], "temperature": 1, "max_tokens": 500}}
```

We begin by importing the relevant packages:

```python
import json
import random
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
```

For this example, we need a persona and activity for which we'll generate the David Attenborough style narration.

[Continue reading the full tutorial on the original post...] 