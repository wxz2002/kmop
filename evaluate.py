import json
from sklearn.metrics import precision_recall_fscore_support
import re


def find_entities(text, entities):
    found_entities = []
    for entity in entities:
        if entity in text:
            found_entities.append((entity, text.index(entity)))
        else:
            found_entities.append((entity, None))  # 如果实体不在文本中，添加None作为索引
    return found_entities

def determine_sentiment(text_segment, positive_words, negative_words):
    if text_segment is None:
        return None  # 没有文本段可分析时返回“无情感”
    sentiment = 0
    if any(word in text_segment for word in positive_words):
        sentiment = 1
    elif any(word in text_segment for word in negative_words):
        sentiment = 2
    return sentiment

def predict_entities_and_sentiments(text, entities):
    found_entities = find_entities(text, entities)
    predictions = []
    for i, entity, start_index in enumerate(found_entities):
        if start_index is not None:
            if i == len(found_entities)-1:
                end_index = len(text)
            else :
                for j in range(i+1, len(found_entities)):
                    if found_entities[j][1] is not None:
                        _, end_index = found_entities[j]
                        break
                    end_index = len(text)
            text_segment = text[start_index, end_index]  # 获取两个实体之间的文本
        else:
            text_segment = None  # 如果没有找到实体，将文本段设为None
        sentiment = determine_sentiment(text_segment)
        predictions.append((entity, sentiment))
    return predictions

if __name__ == "__main__":
    labels = []
    label_path = "./twitter2015/test.json"
    predicts = []
    predict_labels = []
    predict_path = "./twitter2015/test.jsonl"

    label_to_sentiment = {"NEG": 0, "POS": 1, "NEU": 2}

    positive_words = ["positive","Positive","pos","Pos"]
    negative_words = ["negative","Negative","neg","Neg"]

    # 读取标签数据和预测数据
    with open(label_path, "r") as f:
        label_datas = json.load(f)
    with open(predict_path, "r") as f:
        for line in f:
            predict_data = json.loads(line)
            predicts.append(predict_data)

    # 从标签数据中提取实体和情感标签    
    for i, label_data in enumerate(label_datas):
        entities = []
        aspects = label_data["aspects"]
        for aspect in aspects:
            entity = " ".join(aspect["term"])
            polarity = label_to_sentiment[aspect["polarity"]]
            entities.append(entity)
            labels.append((entity, polarity))

        predict_labels += predict_entities_and_sentiments(predicts[i]["answer"], entities)

    # 得到结果
    precision, recall, fscore = precision_recall_fscore_support(labels, predict_labels, average="macro")
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1 Score: {:.2f}".format(fscore))


