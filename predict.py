import os
import argparse
import json
import time
import datetime
from PIL import Image, ImageDraw
import torch
from paddleocr import PaddleOCR


def preprocessing_data_swin(model, data):
    #tokenizer
    encoder_processor = model.encoder.prepare_input
    for sample in data:
        image = Image.open(sample["image_path"])
        encoder_encoding = encoder_processor(image)
        sample["encoding_inputs"] = encoder_encoding
    return data

def preprocessing_data_layoutlm(model, data):
    #tokenizer
    encoder_processor = model.encoder.prepare_input
    for sample in data:
        image = Image.open(sample["image_path"])
        encoder_encoding = encoder_processor(image, sample["texts"], bboxes=sample["boxes"])
        encoder_encoding["bbox"] = (encoder_encoding["bbox"]*1000).to(torch.int32)
        sample["encoding_inputs"] = encoder_encoding
    return data

def visualize_bbox(img, boxes, texts):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for bb, t in zip(boxes, texts):
        x0, y0, x1, y1 = bb
        x0 = x0 * w
        y0 = y0 * h
        x1 = x1 * w
        y1 = y1 * h
        draw.rectangle([x0, y0, x1, y1], fill=None, outline=(225, 0, 100))
        draw.text((x0, y0), t, fill=(225, 0, 225))
    return img

def build_model(model_path):
    with open(f"{model_path}/model_config.json", "r") as f:
        config = json.load(f)
    donut_config = DonutConfig.from_dict(config)
    model = DonutModel(donut_config)
    trained_state_dict = torch.load(f"{model_path}/best_model.cpt")
    model.load_state_dict(trained_state_dict)
    return model


def main(args):
    if args.model_path.split("/")[-1].startswith("swin-en"):
        data = []
        for image_name in os.listdir(args.input_dir):
            image_path = f"{args.input_dir}/{image_name}"
            data.append({"image_path": image_path, "image_name": image_name})
    else:
        print("start ocr")
        ocr = PaddleOCR(use_angle_cls=False, lang='en')
        data = []
        for image_name in os.listdir(args.input_dir):
            image_path = f"{args.input_dir}/{image_name}"
            img = Image.open(image_path)
            w, h = img.size
            results = ocr.ocr(image_path, cls=False)
            result = results[0]
            boxes = []
            texts = []
            if result is not None:
                for line in result:
                    top_left, top_right, bottom_right, bottom_left = line[0]
                    x0, y0 = top_left
                    x1, y1 = bottom_right
                    x0 = x0 / w
                    y0 = y0 / h
                    x1 = x1 / w
                    y1 = y1 / h
                    boxes.append([x0, y0, x1, y1])
                    texts.append(line[1][0])
            draw_img = visualize_bbox(img, boxes, texts)
            draw_img.save(f"{args.output_dir}/ocr_results/{image_name}")
            data.append({"image_path": image_path, "image_name": image_name, "texts": texts, "boxes": boxes})
            
    model = build_model(args.model_path)
    dataset = preprocessing_data(model, data)
    model = model.cuda()
    print("start prediction")
    start_time = time.time()
    if args.model_path.split("/")[-1].startswith("swin-en"):
        generations = []
        for sample in dataset:
            inputs = sample["encoding_inputs"].unsqueeze(0)
            generate_text = model.inference(image_tensors=inputs)["predictions"]
            generations.append({"image_name": sample["image_name"], "generation": generate_text[0]})
    else:
        generations = []
        for sample in dataset:
            inputs = sample["encoding_inputs"]
            generate_text = model.inference(inputs=inputs)["predictions"]
            generations.append({"image_name": sample["image_name"], "generation": generate_text[0]})
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Generation time {}'.format(total_time_str))
    for sample in generations:
      with open(f"{args.output_dir}/model_predictions/{sample['image_name']}.txt", "w") as f:
        f.write(str(sample["generation"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    if args.model_path.split("/")[-1].startswith("swin-en"):
        print("predict by swin-transfomre_encoder&mbart_decoder")
        from donut_swin import DonutConfig, DonutModel
        preprocessing_data = preprocessing_data_swin
        os.makedirs(args.output_dir + "/model_predictions", exist_ok=True)
    else:
        from donut_layoutLMv3_3 import DonutConfig, DonutModel
        print("predict by layoutlmv3_encoder&mbart_decoder")
        preprocessing_data = preprocessing_data_layoutlm
        os.makedirs(args.output_dir + "/ocr_results", exist_ok=True)
        os.makedirs(args.output_dir + "/model_predictions", exist_ok=True)
    
    main(args)
  