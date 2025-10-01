from unsloth import FastModel
import torch
import re
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torchaudio
import io
import os
from datasets import load_from_disk
from datasets import concatenate_datasets
import numpy as np

import requests

def is_arabic(word):
    arabic_count = sum(1 for char in word if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F')
    total_letters = sum(1 for char in word if char.isalpha())
    if total_letters == 0:
        return False
    return arabic_count / total_letters >= 0.5

def process_arabic_sentence(sentence):
    try:
        response = requests.post(
            "http:/localhost/catt",
            json={"text": sentence, "model_type": "Encoder-Only"},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("result", sentence)
    except Exception as e:
        print(f"API error: {e}")
        return sentence 
def process_line(line):
    words = line.strip().split()
    arabic_words = []
    word_positions = []

    # Extract Arabic words and keep track of positions
    for idx, word in enumerate(words):
        if is_arabic(word):
            arabic_words.append(word)
            word_positions.append(idx)

    # Join Arabic words to send to API
    if arabic_words:
        joined_arabic = ' '.join(arabic_words)
        processed_arabic = process_arabic_sentence(joined_arabic)
        processed_arabic_words = processed_arabic.split()
    else:
        processed_arabic_words = []

    # Reconstruct original sentence
    reconstructed = words[:]
    for pos, new_word in zip(word_positions, processed_arabic_words):
        reconstructed[pos] = new_word

    return ' '.join(reconstructed)   

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("MBZUAI/NADI-2025-Sub-task-3-all")
#intial the model
model, processor = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    use_gradient_checkpointing = "unsloth",
    token = "", # use one if using gated models
)
model = model.to("cuda")

#Let's finetune Gemma 3N

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 128,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        # Audio layers
        "post", "linear_start", "linear_end",
        "embedding_projection",
    ],
    modules_to_save=[
        "lm_head",
        "embed_tokens",
        "embed_audio",
    ],
)

def clean_text(text_with_tashkeel):
# Remove tashkeel (harakat)
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670\u06D6-\u06ED]')
    return tashkeel.sub('', text_with_tashkeel).strip()
def format_intersection_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio"])):
        #print(samples['audio'][idx])
        # with io.BytesIO(samples['audio'][idx]['bytes']) as audio_stream:
        #     waveform, sample_rate = torchaudio.load(audio_stream)
        #     samples["audio"][idx]["array"] = torch.from_numpy(waveform)
        
        audio = samples["audio"][idx]["array"].numpy()

        label = str(samples["transcription"][idx])
        clean_label=clean_text(label)
        prompt_user=["قم بالمراجعة للنص مع المحافظة على نفس عدد الكلمات ولا تخرج كلمات جديدة فقط أضف التشكيلات للكلمات العربية: ","","النص",clean_label]
	
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": " أنت مدقق لغوي لديك  ملف  صوتى وترى الكلام المكتوب لتخرج أفضل تَشْكِيل للكلمات العربية فقط وأترك الكلام الغير عربى كما هو كالأنجليزى والفرنسي على سبيل المثال ",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": '\n'.join(prompt_user)}
                ]
            },
            {
                "role": "assistant",
                "content":[{"type": "text", "text": label}]
            }
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples

def format_intersection_data_validation(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"].numpy()
        label = str(samples["transcription"][idx])
        clean_label=clean_text(label)
        prompt_user=["قم بالمراجعة للنص مع المحافظة على نفس عدد الكلمات ولا تخرج كلمات جديدة فقط أضف التشكيلات للكلمات العربية: ","","النص",clean_label]
	
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": " أنت مدقق لغوي لديك  ملف  صوتى وترى الكلام المكتوب لتخرج أفضل تَشْكِيل للكلمات العربية فقط وأترك الكلام الغير عربى كما هو كالأنجليزى والفرنسي على سبيل المثال ",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": '\n'.join(prompt_user)}
                ]
            },
            {
                "role": "assistant",
                "content":[{"type": "text", "text": label}]
            }
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples


def format_intersection_data_tashkeel(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"].numpy()
        label = str(samples["transcription"][idx])
        clean_label=clean_text(label)
        prompt_user=["قم بالمراجعة للنص مع المحافظة على نفس عدد الكلمات ولا تخرج كلمات جديدة فقط أضف التشكيلات للكلمات العربية: ","","النص",clean_label]
        tashkel_catt=process_line(label)
        print(tashkel_catt)
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": " أنت مدقق لغوي لديك  ملف  صوتى وترى الكلام المكتوب لتخرج أفضل تَشْكِيل للكلمات العربية فقط وأترك الكلام الغير عربى كما هو كالأنجليزى والفرنسي على سبيل المثال ",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": '\n'.join(prompt_user)}
                ]
            },
            {
                "role": "assistant",
                "content":[{"type": "text", "text":tashkel_catt }]
            }
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples

def collate_fn(examples):
    texts = []
    audios = []

    for example in examples:
        # Apply chat template to get text
        text = processor.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).strip()
        texts.append(text)

        # Extract audio array
        audio = example["messages"][1]["content"][0]["audio"]  # Directly from message
        audios.append(torch.tensor(audio))

    # Pad audio tensors to the same length
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

    # Convert to list of numpy arrays for processor
    audios = [a.numpy() for a in audios]

    # Tokenize text and audio
    batch = processor(
        text=texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=16000,  # Explicitly specify sampling rate if needed
        max_length=16000*60
    )

    labels = batch["input_ids"].clone()

    # Mask special tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if hasattr(processor.tokenizer, 'image_token_id'):
        labels[labels == processor.tokenizer.image_token_id] = -100
    if hasattr(processor.tokenizer, 'audio_token_id'):
        labels[labels == processor.tokenizer.audio_token_id] = -100
    if hasattr(processor.tokenizer, 'boi_token_id'):
        labels[labels == processor.tokenizer.boi_token_id] = -100
    if hasattr(processor.tokenizer, 'eoi_token_id'):
        labels[labels == processor.tokenizer.eoi_token_id] = -100

    batch["labels"] = labels
    return batch

train_dataset_orig = ds["train"]
train_dataset_orig.set_format(type="torch")
print("Train dataset loaded with {} samples".format(len(train_dataset_orig)))
eval_dataset = ds["dev"]
eval_dataset.set_format(type="torch")
train_dataset_orig = train_dataset_orig.map(format_intersection_data, batched=True, batch_size=4)   
eval_dataset = eval_dataset.map(format_intersection_data_validation, batched=True, batch_size=4)  
train_augmented_dataset = load_from_disk("data/augmentation_nlpaug_new")
augmented_dataset=ds['augment']
augmented_dataset.set_format(type="torch")
augmented_dataset = augmented_dataset.map(format_intersection_data_tashkeel, batched=True, batch_size=4)   
augmented_dataset.save_to_disk("data/augmented_dataset")
augmented_dataset = load_from_disk("data/augmented_dataset")

# Combine original and augmented datasets
train_dataset = concatenate_datasets([train_dataset_orig, train_augmented_dataset, augmented_dataset])
# print(len(train_dataset))
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor.tokenizer,
    data_collator=collate_fn,
    args = SFTConfig(
        per_device_train_batch_size = 5,
        gradient_accumulation_steps = 2,
        # use reentrant checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        warmup_ratio = 0.1,
        # max_steps = 60,
        num_train_epochs = 2,          # Set this instead of max_steps for full training runs
        learning_rate = 5e-5,
        logging_steps = 100,
        save_strategy="steps",
        optim = "adamw_torch",
        weight_decay = 0.01,
        save_total_limit=3,
        # load_best_model_at_end=True,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "lora128_complete_with_augmented_text_checkpoint_16500",
        report_to = "none",             # For Weights and Biases
        # You MUST put the below items for audio finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        # dataset_num_proc = 2,
        max_length = 2048,
    )
)

trainer_stats = trainer.train()
model.save_pretrained("extract_lora128_complete_with_augmented_text_checkpoint_16500")  # Local saving
processor.save_pretrained("extract_lora128_complete_with_augmented_text_checkpoint_16500")