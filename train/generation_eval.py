import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel, AutoTokenizer
from dataset import MMDataset
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch import Tensor
import json
from nltk.translate.bleu_score import sentence_bleu


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(41)

def _compute_entity_f1(preds, refs):
    """Compute Entity-F1."""
    def _count(pred, ref):
        tp, fp, fn = 0, 0, 0
        if len(ref) != 0:
            for g in ref:
                if g in pred:
                    tp += 1
                else:
                    fn += 1
            for p in set(pred):
                if p not in ref:
                    fp += 1
        return tp, fp, fn

    def _count_set(pred, ref):
        predicted_set = set(pred)
        ground_truth_set = set(ref)
        tp = predicted_set & ground_truth_set  # Intersection of predicted and ground truth
        # False Positives: Items predicted but not in ground truth
        fp = predicted_set - ground_truth_set  # Predicted items but not ground truth
        # False Negatives: Items in ground truth but not predicted
        fn = ground_truth_set - predicted_set  # Ground truth items but not predicted
        return len(tp), len(fp), len(fn)

    tp_all, fp_all, fn_all = 0, 0, 0
    for pred, ref in zip(preds, refs):
        tp, fp, fn = _count(pred, ref)
        tp_all += tp
        fp_all += fp
        fn_all += fn

    precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
    recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
    f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    return f1, precision, recall


def encode_text(text, processor, device, model):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)

    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # shape: [1, 512]

def encode_image(image_path, processor, device, model):
    if image_path == "":
        width, height = 800, 600
        image = Image.new('RGB', (width, height), color='white')
    else:
        image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features  # shape: [1, 512]


def compute_prf(gold, pred, kb_plain):
    local_kb_word = [k[2] for k in kb_plain]
    local_kb_word = list(set(local_kb_word))
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            # if p in entities or p in local_kb_word:
            if p in local_kb_word:
                if p not in gold:
                    FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count

def compute_f1(gold, pred):
    pred_set = set(pred)
    gold_set = set(gold)

    tp = pred_set & gold_set
    fp = pred_set - gold_set

    fn = gold_set - pred_set

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1

def extract_entity_from_response(knowledge, response):
    entities = set()

    for entry in knowledge:
            
        for key, value in entry.items():
            if not isinstance(value, list):
                if value in response:
                    entities.add(value)

    unique_entities = list(entities)
    
    return unique_entities

def bleu_update(hypothesis, reference):
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()

    bleu = sentence_bleu([ref_tokens], hyp_tokens)

    return bleu

def clean_text(text):
    text = text.replace("."," .").replace("'"," ' ").replace("?"," ?").replace(","," ,").replace("!"," !").replace("  "," ").replace("you are","you ' re")
    return text

def my_eval(cl_clipPre_model, flanT5_model, fusion_model, mapper, processor, tokenizer, device):
    with open("../../../data/m-rest/used/test_label.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_data = []
    bleu_list = []

    score = 0
    count = 0
    acc = 0
    image_response_acc = 0

    preds = []
    refs = []

    iter = tqdm(data)
    for index, item in enumerate(iter):
        image_context = ""
        text_context = ""
        str = ""
        for i, utt in enumerate(item["history"]):
            if text_context == "":
                text_context = utt
            else:
                text_context += " " + utt
            if i == len(item["history"])-1:
                user_question = utt
                
            if i % 2 == 0:
                if str == "":
                    str += "<user> " + utt
                else:
                    str += " <user> " + utt
            else:
                str += " <system> " + utt
            
        context_token = str

        image_response = item["image"]

        text_context_features = encode_text(text_context, processor, device, cl_clipPre_model)
        image_context_features = encode_image(image_context, processor, device, cl_clipPre_model)

        fused_context_representation = fusion_model(text_context_features, image_context_features)
        # fused_context_representation = text_context_features

        image_attraction_rep_list = []
        image_attraction_list = []

        similarites = []

        fused_entity_rep_list = []

        available_entities = []

        white_entity = {
            "name": "",
            "area": "",
            "food": "",
            "phone": "",
            "pricerange": "",
            "location": "",
            "address": "",
            "type": "",
            "id": "",
            "postcode": "",
            "bar": "",
            "garden": "",
            "dining_room": "",
            "label": ""
        }

        all_knowledge = item["knowledge"]
        all_knowledge.append(white_entity)

        for knowledge in all_knowledge:
            fused_entity_rep = None

            if knowledge["label"] == "1":
                available_entities.append(knowledge)

                # knowledge.pop("label")

            for kk, vv in knowledge.items():
                if kk == "label":
                    continue
                elif "png" in vv or "jpg" in vv or "jpeg" in vv:
                    text_kk_feat = encode_text(kk, processor, device, cl_clipPre_model)
                    image_vv_feat = encode_image("../../../data/m-rest/images/" + vv , processor, device, cl_clipPre_model)
                    fused_attraction_rep = fusion_model(text_kk_feat, image_vv_feat)
                elif isinstance(vv, list):
                    text_kk_feat = encode_text(kk, processor, device, cl_clipPre_model)
                    fused_list_rep = None
                    for img in vv:
                        image_vv_feat = encode_image("../../../data/m-rest/images/" + img , processor, device, cl_clipPre_model)
                        fused_img_rep = fusion_model(text_kk_feat, image_vv_feat)

                        if fused_list_rep is None:
                            fused_list_rep = fused_img_rep
                        else:
                            fused_list_rep = torch.cat([fused_list_rep, fused_img_rep], dim=0)
                    fused_attraction_rep = fused_list_rep
                else:
                    fused_attraction_rep = encode_text(kk + ": " + vv, processor, device, cl_clipPre_model)

                if fused_entity_rep is None:
                    fused_entity_rep = fused_attraction_rep
                else:
                    fused_entity_rep = torch.cat([fused_entity_rep, fused_attraction_rep], dim=0)
            
            fused_entity_rep_list.append(fused_entity_rep)


            similarity = F.cosine_similarity(fused_context_representation, fused_entity_rep)
            similarites.append(similarity.mean())

        similarity_values = [similarity.item() for similarity in similarites]

        most_similar_idx = np.argmax(similarity_values)
        most_similar_entity = all_knowledge[most_similar_idx]

        select_pos_image_entity = dict()
        if most_similar_entity["name"] == "":
            select_pos_image_entity = all_knowledge[1]
        else:
            select_pos_image_entity = most_similar_entity

        for kkk, vvv in select_pos_image_entity.items():
            if "png" in vvv or "jpg" in vvv or "jpeg" in vvv :
                image_attraction_rep = encode_image("../../../data/m-rest/images/" + vvv , processor, device, cl_clipPre_model)
                image_attraction_rep_list.append(image_attraction_rep)
                image_attraction_list.append(vvv)
            elif isinstance(vvv, list):
                for dish_image in vvv:
                    image_attraction_rep = encode_image("../../../data/m-rest/images/" + dish_image , processor, device, cl_clipPre_model)
                    image_attraction_rep_list.append(image_attraction_rep)
                    image_attraction_list.append(dish_image)
        
        white_image = ""
        white_image_rep = encode_image(white_image, processor, device, cl_clipPre_model)
        image_attraction_rep_list.append(white_image_rep)
        image_attraction_list.append(white_image)

        user_question_features = encode_text(user_question, processor, device, cl_clipPre_model)
        image_context_features = encode_image(image_context, processor, device, cl_clipPre_model)

        fused_user_question_representation = fusion_model(user_question_features, image_context_features)
        # fused_user_question_representation = user_question_features

        similarites_image = []
        for every_image_attraction in image_attraction_rep_list:
            similarity_image = F.cosine_similarity(fused_user_question_representation , every_image_attraction)
            similarites_image.append(similarity_image.mean())
                
        similarity_values_image = [similarity.item() for similarity in similarites_image]

        most_similar_idx_image = np.argmax(similarity_values_image)
        most_similar_image = image_attraction_list[most_similar_idx_image]

        if most_similar_image != "":
            most_similar_image = "../../../data/m-rest/images/" + most_similar_image


        inputs_context = tokenizer(context_token, return_tensors='pt', truncation=True, padding='max_length', max_length=328)
        inputs_context_ids = {key: value.to(device) for key, value in inputs_context.items()}

        input_ids_ctx = inputs_context_ids["input_ids"]

        embedding_ctx = flanT5_model.get_input_embeddings()(input_ids_ctx)
 
        image_ctx_features = encode_image(image_context , processor, device, cl_clipPre_model)

        mapped_image_context_features = mapper(image_ctx_features)

        full_context_input = torch.cat([embedding_ctx, mapped_image_context_features], dim=1).to(device)
        # full_context_input = embedding_ctx.to(device)
        
        text = "; ".join(f"{k}: {v}" for k, v in most_similar_entity.items())
        inputs_entity = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=328)
        inputs_entity_ids = {key: value.to(device) for key, value in inputs_entity.items()}

        input_ids_entity = inputs_entity_ids["input_ids"]

        embedding_entity = flanT5_model.get_input_embeddings()(input_ids_entity)

        image_response_features = encode_image(most_similar_image , processor, device, cl_clipPre_model)

        mapped_image_response_features = mapper(image_response_features)

        full_entity_input = torch.cat([embedding_entity, mapped_image_response_features], dim=1).to(device)

        full_input = torch.cat([full_context_input, full_entity_input], dim=1).to(device)

        output_ids = flanT5_model.generate(
            inputs_embeds=full_input,
            max_length=64
        )

        generated_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        gold_entities = extract_entity_from_response(item["knowledge"], item["response"])
        pred_entities = extract_entity_from_response(item["knowledge"], generated_response)

        preds.append(pred_entities)
        refs.append(gold_entities)

        triples = []
        for record in item["knowledge"]:
            entity = record["name"]
            for key, value in record.items():
                if not isinstance(value, list):
                    if key != "name":
                        triples.append((entity, key, value))

        f1, _ = compute_prf(gold_entities, generated_response, triples)

        if (len(gold_entities) > 0):
            count += 1
        
        score += f1

        bleu = bleu_update(clean_text(generated_response), clean_text(item["response"])) 
        bleu_list.append(bleu)

        if most_similar_image != "":
            final_most_similar_image_list = most_similar_image.split("/")[-2:]
            final_most_similar_image = "/".join(final_most_similar_image_list)
        else:
            final_most_similar_image = ""

        output_data.append({
            'gold_response': item["response"],
            'generated_response': generated_response,
            'gold_image': image_response,
            'generated_image': final_most_similar_image,
            'most_similar_entity': most_similar_entity
        })

        if len(available_entities) != 0:
            for av_entity in available_entities:
                if most_similar_entity["name"] == av_entity["name"]:
                    acc += 1
                    break
        else:
            if most_similar_entity["name"] == "":
                acc += 1

        if final_most_similar_image == image_response:
            image_response_acc += 1


    avg_f1 = score/(count+1e-30)

    entity_f1, entity_precision, entity_recall = _compute_entity_f1(preds, refs)

    avg_bleu = sum(bleu_list) / len(bleu_list)

    print("f1:", avg_f1)
    print("entity_f1:", entity_f1)
    print("bleu:", avg_bleu)

    entity_acc_rate = acc / len(data)

    final_image_response_acc = image_response_acc / len(data)

    print("entity_acc_rate:", entity_acc_rate)
    print("final_image_response_acc:", final_image_response_acc)

    return output_data
    