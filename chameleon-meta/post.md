# Notes for "Chameleon: Mixed-Modal Early-Fusion Foundation Models"

> Here is the [link](https://arxiv.org/abs/2405.09818) to the original paper. These notes were prepared for my [LLM Asia Paper Club](https://lu.ma/dcogesvb) sharing. Any feedback or areas for improvement would be most appreciated at _cyzgab[at]gmail.com_.

# Key Points

1. End-to-end multimodal tokens (i.e. no modality-specific encoder or decoder)

2. Novelties in architectural innovations and training techniques to address computational challenges:
    a. query-key normalization
    b. revised placement of layer norms

3. Pre-training these models require large datasets and computation. 
    a. Dataset of 4.4T (2.9T is text only, 1.5T is text-to-image, and 400B is text-and-image-interleaved)
    b. 7B and 34B trained on 856K and 4.2M GPU hours respectively

4. Performance:
    a. On Visual Q&A, outperforms Flamingo, Llava-1.5
    b. On text-only benchmarks, still competitive with Mixtral 8x7B and Gemini-Pro
    c. On human pairwise comparisons, beats Gemini-Pro and GPT-4V

___

![**Image 1:** Conceptual summary of multimodal training and generation from the paper](thumbnail.jpg)

___

![**Image 2:** Example generation from the paper](sample_generation.png)

___

# Late vs Early Fusion

> A useful reference for me was this [literature review](https://arxiv.org/abs/2405.17927) by Wadekar et.al

**Late fusion** Done at the internal layers of the model (e.g. OpenFlamingo, LLaMA-Adapter-V2)

___

![**Image 3:** Simplified architectural summary for late fusion - taken from Wadekar et.al](late_fusion_example.png)

___


**Early fusion** Done at the input stage (e.g. LLaVA, Unified-IO-2, Chameleon, Gemini)

___

![**Image 4:** Simplified architectural summary for non-tokenised early fusion - taken from Wadekar et.al](early_fusion_encoder.png)


___

![**Image 5:** Simplified architectural summary for tokenised early fusion - taken from Wadekar et.al](early_fusion_tokens.png)

___


# Tokeniser

- For images: trained a new image tokenizer based on [Gafni et.al (2022)](https://arxiv.org/abs/2203.13131) which encodes a 512 × 512 image into 1024 discrete tokens from a codebook of size 8192

> Based on [OAI's pricing page](https://openai.com/api/pricing/) (as of 5 June 2024), one image is ~170 tokens in GPT-4o.

- For text: BPE tokenizer over a subset of training data, with a vocabulary size of 65K

# Ensuring Stability of Pre-Training

> _"We found that the standard LLaMa architecture showed complex divergences due to slow norm growth in the mid-to-late stages of training. We narrowed down the cause of the divergence to the softmax operation being problematic when training with multiple modalities of significantly varying entropy due to the translation invariant property of softmax (i.e., softmax(z) = softmax(z+c)). Because we share all weights of the model across modalities, each modality will try to "compete" with the other by increasing its norms slightly; while not problematic at the beginning of training, it manifests in divergences once we get outside the effective representation range of bf16... In a unimodal setting, this problem has also been named the logit drift problem." (Page 6)_

1. Query-key normalisation: applying layer norm to the query and key vectors within the attention.

2. Revised placement of layer norms for 34B model 

3. Introducing z-loss regularisation 

___

![**Image 6:** Training plots](training_plots.png)

___


Dropout was initially introduced after the attention and feed forward layer for the 7B model, though subsequently found to be not necessary. For the 34B model, dropout was not sufficient (nor necessary). 

# Summary of Pre-Training

| Model      | Params | Context Length | GQA | Tokens | LR            | Epochs | Dropout | Zloss  | Qknorm |
|------------|--------|----------------|-----|--------|---------------|--------|---------|--------|--------|
| LLMa-1     | 7B     | 2k             | ✗   | 1.0T   | 3.0 × 10^-4   | 1.0    | 0.0     | 0.0    | ✗      |
|            | 33B    | 2k             | ✗   | 1.4T   | 1.5 × 10^-4   | 1.0    | 0.0     | 0.0    | ✗      |
| LLMa-2     | 7B     | 4k             | ✗   | 2.0T   | 3.0 × 10^-4   | 1.0    | 0.0     | 0.0    | ✗      |
|            | 34B    | 4k             | ✓   | 2.0T   | 1.5 × 10^-4   | 1.0    | 0.0     | 0.0    | ✗      |
| Chameleon  | 7B     | 4k             | ✗   | 4.4T   | 1.0 × 10^-4   | 2.1    | 0.1     | 10^-5  | ✓      |
|            | 34B    | 4k             | ✓   | 4.4T   | 1.0 × 10^-4   | 2.1    | 0.0     | 10^-5  | ✓      |

_Taken from Table 1 of the paper_


# Challenges Associated with Inference

1. When decoding, we need to check whether it is a text or image token
2. Masking tokens from other modalities when exclusively generating for a particular modality (e.g. no text tokens when doing image-only generation)
3. Token-based image generation is a fixed-sized block

# SFT/Alignment

Supervised fine-tuning dataset covered the following categories:

- text
- code
- visual chat
- image generation
- interleaved text/image generation
- safety (e.g. "I can't help with that") 