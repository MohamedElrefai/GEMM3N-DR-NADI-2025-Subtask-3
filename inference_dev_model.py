from unsloth import FastModel
from datasets import load_dataset
import requests
import re
from tqdm import tqdm
import numpy as np
from jiwer import wer  # pip install jiwer
import csv
import requests
# Load model
model, processor = FastModel.from_pretrained(
    model_name="starting_from_outputs_training_text_only_epoch2_lora128_complete_with_augmented_text/checkpoint-16500",
    max_seq_length=1024,
    load_in_4bit=False,
)
model = model.to("cuda")

# Arabic checker
def is_arabic(word):
    arabic_count = sum(1 for char in word if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F')
    total_letters = sum(1 for char in word if char.isalpha())
    return arabic_count / total_letters >= 0.5 if total_letters > 0 else False

# Remove diacritics
def clean_text(text_with_tashkeel):
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670\u06D6-\u06ED]')
    return tashkeel.sub('', text_with_tashkeel).strip()

# Process one sample
def process_arabic_sentence(text, audio, temperature,duration_seconds):
    prompt_user = [
       "رجاءً أضف التشكيل لكل حرف من الحروف العربية في الجملة التالية:" , 
        "",
        "أمثلة:",
    "مثال 1:",
    "ذهب محمد إلى المدرسة",
    "ذَهَبَ مُحَمَّدٌ إِلَى الْمَدْرَسَةِ",
    "",
    "مثال 2:",
    "معلش يا حبيبي، بكرة تبقى أحسن",
    "مَعْلِشّ يَا حَبِيبِي، بُكْرَة تِبْقَى أَحْسَن",
    "",
    "مثال 3:",
    "إنت لسه صاحي؟ دا الدنيا نهار",
    "إِنْتَ لِسَّه صَاحِي؟ دَا الدُّنْيَا نَهَار",
    "",
     "مثال 4:",
     "الجو سخون بزاف",
     "الجُّو سْخُونْ بْزَّاف",
         "",
      "مثال 5:",
     "الولد جالس في الحوش",
     "الْوَلَد جَالِسْ فِي الْحَوْش",
         "",
     "مثال 6:",
     "لوس أنجلس",
     "لُوسْ أَنْجِلِسْ",
         "",
     "مثال 7:",
     "ماذا تريد أن تفعل؟",
     "مَاذَا تُرِيدُ أَنْ تَفْعَلَ؟",
         "",
     "النص:",
        text,
    ]

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "أنت مدقق لغوي، لديك ملف صوتي والحروف المكتوبة، أخرج التشكيل الأمثل لكافة الحروف العربية .",
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": '\n'.join(prompt_user)},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
        truncation=True,
        max_length=int(16000 *duration_seconds),
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=temperature,
        do_sample=False,
        top_k = 1,
        top_p = 1.0
    )

    decoded = processor.decode(outputs[0], skip_special_tokens=True)
    return decoded.split('model')[-1].strip()

def process_segment(line, audio, temperature,duration_seconds):
    words = line.strip().split()
    arabic_words = []
    word_positions = []

    for idx, word in enumerate(words):
        if is_arabic(word):
            arabic_words.append(word)
            word_positions.append(idx)

    if arabic_words:
        joined_arabic = ' '.join(arabic_words)
        print(joined_arabic)
        processed_arabic = process_arabic_sentence(joined_arabic, audio, temperature,duration_seconds)
        processed_arabic_words = processed_arabic.split()
    else:
        processed_arabic_words = []

    reconstructed = words[:]
    for pos, new_word in zip(word_positions, processed_arabic_words):
        reconstructed[pos] = new_word

    return ' '.join(reconstructed)

# Load dataset
ds = load_dataset("MBZUAI/NADI-2025-Sub-task-3-all")
eval_dataset = ds["dev"]
# eval_dataset = eval_dataset.select(range(10))  # Optional: use smaller subset for debugging

# Temperatures to test
temperatures = [0.001]
best_wer = float("inf")
best_temp = None

# Loop through temperatures
for temp in temperatures:
    refs = []
    hyps = []
    csv_file = f"results_temp_{temp:.3f}_with_catt_inference.csv"
    txt_file = f"wer_temp_{temp:.3f}_with_catt_inference.txt"

    print(f"\n🔁 Running inference at temperature {temp:.3f}")

    with open(csv_file, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Reference", "Prediction", "Input"])

        for sample in tqdm(eval_dataset, desc=f"Temp {temp:.2f}"):
            # sample=eval_dataset[1516]
            ref = sample["transcription"]
            input_text = clean_text(ref)
            audio = sample["audio"]['array']
            duration_seconds = min(len(audio) / 16000,30)
            print(duration_seconds)
            print(len(input_text))
            # print(duration_seconds)
            # break
            pred = process_segment(input_text, audio, temp,duration_seconds)

            refs.append(ref)
            hyps.append(pred)
            writer.writerow([ref, pred, input_text])

    # Compute WER
    final_wer = wer(refs, hyps)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Temperature: {temp:.2f}\nFinal WER: {final_wer:.4f}\n")

    print(f"✅ Temp {temp:.3f} -> WER: {final_wer:.4f}")

    if final_wer < best_wer:
        best_wer = final_wer
        best_temp = temp

print(f"\n🏆 Best temperature: {best_temp:.2f} with WER: {best_wer:.4f}")
