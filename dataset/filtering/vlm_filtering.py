import os
import json
import torch
import torch.multiprocessing as mp
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import logging

# --- 설정 ---
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
GROUPED_DATA_PATH = "/data1/temp_0306/duplicates/annotation_mismatch.json"
MANIFEST_PATH = "/data1/temp_0306/manifest.jsonl"
OUTPUT_DIR = "/data1/qwen2.5-vl/filtering_output"
CROPPED_IMG_DIR = "/data1/qwen2.5-vl/cropped_images"
BASE_IMAGE_ROOT = "/data1/temp_0306" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPPED_IMG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_bbox_map(manifest_path):
    bbox_map = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            filename = os.path.basename(data["image_path"])
            if filename not in bbox_map:
                bbox_map[filename] = []
            bbox_map[filename].append({"text": data["text"], "bbox": data["bbox"]})
    return bbox_map

def worker(rank, data_chunk, bbox_map, output_file):
    torch.cuda.set_device(rank)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=f"cuda:{rank}"
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"

    results = []
    
    for group in data_chunk:
        try:
            # 1. 모든 후보 이미지 Crop 준비
            crop_images = []
            candidate_list = []
            
            for filename, ann_text in zip(group['files'], group['annotations']):
                clean_text = ann_text.strip("[]'\"")
                img_path = os.path.join(BASE_IMAGE_ROOT, "[원천]Training_간판_가로형간판_원천데이터1", filename)
                
                bbox_entries = bbox_map.get(filename, [])
                entry = next((e for e in bbox_entries if e['text'] == clean_text), None)
                bbox = entry['bbox'] if entry else [0, 0, 1600, 1200]
                
                img = Image.open(img_path).convert("RGB")
                cropped_img = img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
                save_path = os.path.join(CROPPED_IMG_DIR, f"{group['hash']}_{filename}")
                cropped_img.save(save_path, "JPEG")
                
                crop_images.append(cropped_img)
                candidate_list.append(clean_text)

            # 2. 통합 프롬프트 작성 (모델에게 후보군 선택 강제)
            candidates_text = "\n".join([f"- 후보 {i+1}: {text}" for i, text in enumerate(candidate_list)])
            prompt = f"다음은 동일한 간판을 찍은 이미지들입니다. 이 중 가장 정확한 텍스트를 선택하세요.\n{candidates_text}\n\n출력 형식: {{\"best_text\": \"...\", \"reasoning\": \"...\"}}"
            
            # 3. 모델 입력 구성 (모든 이미지를 한 메시지에 전달)
            content = []
            for img in crop_images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            # 4. 추론
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            response = processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)], skip_special_tokens=True)[0]
            
            # 5. 결과 파싱
            match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            decision = json.loads(match.group(0)) if match else {"best_text": "ERROR", "reasoning": response}
            
            results.append({"hash": group["hash"], "decision": decision, "candidates": candidate_list})
            
        except Exception as e:
            logging.error(f"GPU {rank} 처리 에러 ({group['hash']}): {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results: f.write(json.dumps(res, ensure_ascii=False) + '\n')
    logging.info(f"GPU {rank} 작업 완료.")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    with open(GROUPED_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    bbox_map = load_bbox_map(MANIFEST_PATH)
    
    gpu_ids = [1, 2, 3]
    num_gpus = len(gpu_ids)
    chunk_size = len(all_data) // num_gpus
    processes = []
    
    for i, gpu_id in enumerate(gpu_ids):
        data_chunk = all_data[i * chunk_size : (i + 1) * chunk_size] if i != num_gpus - 1 else all_data[i * chunk_size :]
        output_file = os.path.join(OUTPUT_DIR, f"result_gpu_{gpu_id}.jsonl")
        p = mp.Process(target=worker, args=(gpu_id, data_chunk, bbox_map, output_file))
        p.start()
        processes.append(p)
    for p in processes: p.join()