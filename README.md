# Indian Penal Code Legal Assistant - Fine-tuned Language Model

## Project Overview

This project presents a specialized legal assistant based on the Qwen 2.5 1.5B Instruct model, fine-tuned on Indian Penal Code (IPC) sections. The model is designed to provide accurate, comprehensive information about Indian criminal law, serving as an educational and reference tool for legal professionals, students, and citizens.


https://colab.research.google.com/drive/12CvvlXuLs9ITIDjMSTrWTltLqQwEjvhR?authuser=1#scrollTo=TdyUDKq-h8Rx


## Model Architecture

**Base Model:** Qwen/Qwen2.5-1.5B-Instruct

**Fine-tuning Method:** Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA)

**Model Size:** 1.5 billion parameters (3.09 GB)

**Trainable Parameters:** 18,464,768 (1.18% of total parameters)

## Dataset

**Source:** karan842/ipc-sections (Hugging Face Datasets)

**Dataset Statistics:**
- Total IPC Sections: 444
- Cleaned Sections: 441
- Training Examples Generated: 2,205
- Training Split: 1,984 examples (85%)
- Validation Split: 221 examples (15%)

**Data Augmentation Strategy:**
Each IPC section was converted into 5 distinct question-answer pairs to enhance model learning diversity:
1. Direct section query
2. Legal definition query
3. Section explanation query
4. Punishment-specific query
5. Offense-specific query

## Training Configuration

### LoRA Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.1 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Bias | none |
| Task Type | Causal Language Modeling |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Number of Epochs | 3 |
| Batch Size (per device) | 2 |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 16 |
| Learning Rate | 1e-5 |
| Learning Rate Scheduler | Linear with warmup |
| Warmup Steps | 200 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Precision | FP16 (Mixed Precision) |
| Max Sequence Length | 1024 tokens |

### Training Infrastructure

- **Hardware:** NVIDIA T4 GPU (Google Colab)
- **Memory Optimization:** Gradient checkpointing enabled
- **Training Duration:** Approximately 2-4 hours
- **Total Training Steps:** 372
- **Evaluation Strategy:** Every 250 steps
- **Checkpoint Strategy:** Save best model based on validation loss

## Training Results

### Loss Metrics

| Metric | Value |
|--------|-------|
| Training Loss (Step 250) | 0.6908 |
| Validation Loss (Step 250) | 0.6653 |
| Best Validation Loss | 0.6653 |

The consistent decrease in both training and validation loss indicates effective learning without overfitting.

### Weights & Biases Dashboard

Complete training metrics, visualizations, and experiment tracking are available on Weights & Biases:

**W&B Project:** [View Training Dashboard](https://wandb.ai/khanum-z-sabiha-no/huggingface)

**W&B Run:** [usual-durian-4](https://wandb.ai/khanum-z-sabiha-no/huggingface?nw=nwuserkhanumzsabiha)

#### Training Visualizations

The W&B dashboard includes:
- Real-time training and validation loss curves
- Learning rate scheduling visualization
- GPU utilization and memory usage
- Step-by-step training metrics
- Model checkpoint comparisons

**Training Loss Curve:**

![Training Loss]

<img width="466" height="382" alt="image" src="https://github.com/user-attachments/assets/5446baa1-6d48-41c6-a386-db5f629a4c34" />


**Validation Loss Curve:**

![Validation Loss]

<img width="467" height="378" alt="image" src="https://github.com/user-attachments/assets/ab828d79-f265-4d02-94bf-e236d1bb0873" />


**Learning Rate Schedule:**

![Learning Rate]

<img width="1043" height="406" alt="image" src="https://github.com/user-attachments/assets/8dbdeaa9-98bb-425c-be18-70ae497fe8a8" />


> Note: If the above images are not visible, please access the complete interactive dashboard through the W&B project link above.

## Model Capabilities

The fine-tuned model is capable of:

1. **Section Identification:** Accurately identifying and explaining specific IPC sections
2. **Legal Definition Retrieval:** Providing precise legal definitions and terminology
3. **Punishment Details:** Specifying applicable punishments for various offenses
4. **Offense Classification:** Categorizing and describing criminal offenses
5. **Legal Context:** Providing contextual information about Indian criminal law

## Evaluation Methodology

### Qualitative Evaluation Framework

The model was evaluated using a comprehensive qualitative assessment framework with the following criteria:

1. **Coverage Score:** Percentage of expected legal key points mentioned in the response
2. **Quality Rating:** Categorical assessment (Excellent/Good/Fair/Poor)
3. **Completeness:** Response length and detail level
4. **Legal Disclaimer Presence:** Verification of appropriate legal advisory warnings

### Test Dataset Composition

15 test cases across multiple categories:
- Serious Crimes (Murder, Homicide)
- Sexual Offenses
- Property Crimes (Theft, Cheating, Breach of Trust)
- Assault and Violence
- Women's Safety
- General Legal Understanding

Test cases vary in complexity (Low, Medium, High) to assess model performance across different query types.


## Implementation Details

### Technology Stack

- **Framework:** PyTorch, Hugging Face Transformers
- **Fine-tuning Library:** PEFT (Parameter-Efficient Fine-Tuning)
- **Quantization:** bitsandbytes
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Development Environment:** Google Colab with GPU acceleration

### Model Loading and Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = ""

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

conversation = [
    {"role": "system", "content": "You are an expert legal assistant specializing in Indian Penal Code."},
    {"role": "user", "content": "What is Section 302 of the Indian Penal Code?"}
]

prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations and Disclaimers


 **Limited Scope:** The model is trained specifically on IPC sections and may not have comprehensive knowledge of amendments, case law, or procedural aspects.

 **No Legal Authority:** Responses generated by this model do not constitute legal advice and should not be used as the sole basis for legal decisions.

 **Consultation Recommended:** Users should always consult qualified legal professionals for specific legal matters.

 **Model Limitations:** As with all language models, responses may occasionally contain inaccuracies or incomplete information.

## Use Cases

- Legal education and academic research
- Quick reference for IPC sections
- Preliminary legal information gathering
- Law student study assistance
- Public legal awareness initiatives

## Future Improvements

1. Expansion to include more comprehensive IPC sections
2. Integration of recent amendments and updates to the IPC
3. Addition of case law references
4. Multi-turn conversation capability for complex legal queries
5. Integration with other Indian legal codes (CrPC, Evidence Act)

## Repository Structure

```
ipc-legal-qwen/
├── trained_model/
│   └── final/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
├── evaluation_report.txt
├── evaluation_report.json
├── training_script.py
└── README.md
```



## License

This project is released under the MIT License, consistent with the base Qwen model licensing.

## Acknowledgments

- Alibaba Cloud for the Qwen 2.5 base model
- Hugging Face for the transformers library and model hosting
- karan842 for the IPC sections dataset
- Google Colab for providing GPU resources


**Version:** 1.0  
**Last Updated:** October 2025  
**Model Release Date:** October 2025
