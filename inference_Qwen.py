import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import json

### CURRENTLY FORCING MODEL TO RUN ON GPU 0 ###

def main():
    # Configuration #
    test_dir = "./data/test"
    output_path = "./Qwen_outputs.json"

    # Load Model and Processor 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,    ## Changed from "auto"
        # attn_implementation="flash_attention_2",
        device_map={"":0}     ##"auto"
    )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

    # Open dictionary of test labels #
    with open("test_labels_dict.json", "r") as f:
        label_dict = json.load(f)

    # Inference Loop #
    output_dict = {}

    # List of the files in the test directory
    image_files = [f for f in os.listdir(test_dir) if f.endswith(".png")]
    image_files_subset = image_files[:3]

    for fname in tqdm(image_files_subset):
        # Adds ./data/test to start of file name "/fname"
        path = os.path.join(test_dir, fname)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image":path,},
                    {"type": "text", "text": "As an expert radiologist, please describe this chest x-ray."},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0") ##inputs.to("cuda")

        # Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
        output_text = output_text[0]
        print(output_text)

        # Associates each fname to a text:label dictionary
        output_dict[fname] = {"text":output_text, "true_label":label_dict.get(fname)}

    # Save Results #
    with open(output_path, "w") as f:
        print(f"Final output_dict has {len(output_dict)} entries.")
        json.dump(output_dict, f, indent=2)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
