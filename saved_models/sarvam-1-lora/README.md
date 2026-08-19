---
base_model: sarvamai/sarvam-1
library_name: peft
model_name: sarvam-1-lora
tags:
- base_model:adapter:sarvamai/sarvam-1
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for sarvam-1-lora

This model is a fine-tuned version of [sarvamai/sarvam-1](https://huggingface.co/sarvamai/sarvam-1).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- PEFT 0.20.0
- TRL: 1.10.0
- Transformers: 5.15.0
- Pytorch: 2.12.0.dev20260408+cu128
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and GallouÃ©dec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```