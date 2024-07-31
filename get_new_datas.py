import json
import os

def get_new_test_datas(caption_path, original_path, output_path):
    with open(caption_path, 'r') as f:
        lines = f.readlines()
    caption_datas = []
    for line in lines:
        caption_data = json.loads(line)
        caption_datas.append(caption_data)

    with open(original_path, 'r') as f:
        original_datas = json.load(f)

    if len(caption_datas) != len(original_datas):
        raise ValueError("The number of datas in caption file and original file is not the same.")
    
    new_test_datas = []
    for i in range(len(caption_datas)):
        new_test_data = {
            "question_id": i+1,
            "image": original_datas[i]['image_id'],
            'text': "context:{}\ncaption:{}\nBased on the provided image and context, extract aspect terms and predict sentiment.".format(" ".join(original_datas[i]['words'], caption_datas[i]['text'])),
        }
        new_test_datas.append(new_test_data)
    with open(output_path, 'w') as f:
        for data in new_test_datas:
            f.write(json.dumps(data) + "\n")

def get_new_train_datas(caption_path, original_path, dataset, mode, withcot, answer_format):
    with open(caption_path, 'r') as f:
        lines = f.readlines()
    caption_datas = []
    for line in lines:
        caption_data = json.loads(line)
        caption_datas.append(caption_data)

    with open(original_path, 'r') as f:
        original_datas = json.load(f)

    if len(caption_datas) != len(original_datas):
        raise ValueError("The number of datas in caption file and original file is not the same.")
    
    new_llava_datas = []

    id_to_sentiment = {
        "POS": "positive",
        "NEG": "negative",
        "NEU": "neutral"
    }
    
    for i, data in enumerate(original_datas):
        if withcot=="no_cot":
            conversations = []
            context = " ".join(data['words'])
            question = "context:{}\ncaption:{}\nBased on the provided image and context, extract aspect terms and predict sentiment.".format(context, caption_datas[i]['text'])
            conversations.append({
                "from": "human",
                "value": "<image>/nas_mm_2/fangly.fly/data/test/MABSA/{}</image>\n{}".format(data['image_id'],question)
            })
            answers = []
            if answer_format == "format1":
                for aspect in data['aspects']:
                    answers.append("{" + " ".join(aspect['term']) + ": " + id_to_sentiment[aspect['polarity']] + "}")
            elif answer_format == "format2":
                for aspect in data['aspects']:
                    answers.append("the sentiment of {} is {}".format(" ".join(aspect['term']), id_to_sentiment[aspect['polarity']]))
            conversations.append({
                "from": "gpt",
                "value": ", ".join(answers)
            })
            new_llava_datas.append({
                "uid": i+1,
                "image": "/nas_mm_2/fangly.fly/data/test/MABSA/{}".format(data['image_id']),
                "conversations": conversations,
                "source": "Twitter-15" if dataset == "twitter15" else "Twitter-17"
            })
            continue
        elif withcot=="with_cot":
            conversations = []
            context = " ".join(data['words'])
            question = "context:{}\ncaption:{}\nBased on the provided image and context, extract aspect terms.".format(context, caption_datas[i]['text'])
            conversations.append({
                "from": "human",
                "value": "<image>/nas_mm_2/fangly.fly/data/test/MABSA/{}</image>\n{}".format(data['image_id'],question)
            })
            entitys = []
            for aspect in data['aspects']:
                entitys.append(" ".join(aspect['term']))
            conversations.append({
                "from": "gpt",
                "value": "[" + ", ".join(entitys) + "]"
            })
            new_llava_datas.append({
                "uid": i+1,
                "image": "/nas_mm_2/fangly.fly/data/test/MABSA/{}".format(data['image_id']),
                "conversations": conversations,
                "source": "Twitter-15" if dataset == "twitter15" else "Twitter-17"
            })
            for aspect in data['aspects']:
                entity = " ".join(aspect['term'])
                conversations = []
                context = " ".join(data['words'])
                question = "context:{}\ncaption:{}\nBased on the provided image and context, predict the sentiment of {}".format(context, caption_datas[i]['text'], entity)
                conversations.append({
                    "from": "human",
                    "value": "<image>/nas_mm_2/fangly.fly/data/test/MABSA/{}</image>\n{}".format(data['image_id'],question)
                })
                conversations.append({
                    "from": "gpt",
                    "value": "The sentiment of {} is {}".format(entity, id_to_sentiment[aspect['polarity']])
                })
                new_llava_datas.append({
                    "uid": i+1,
                    "image": "/nas_mm_2/fangly.fly/data/test/MABSA/{}".format(data['image_id']),
                    "conversations": conversations,
                    "source": "Twitter-15" if dataset == "twitter15" else "Twitter-17"
                })
            continue
    if not os.path.exists('./new_LLava_data/{}'.format(dataset)):
        os.makedirs('./new_LLava_data/{}'.format(dataset))
    if withcot=="no_cot":
        with open('./new_LLava_data/{}/{}_{}_{}.jsonl'.format(dataset, mode, withcot, answer_format), 'w') as f:
            for conversations in new_llava_datas:
                f.write(json.dumps(conversations) + "\n")
    elif withcot=="with_cot":
        with open('./new_LLava_data/{}/{}_{}.jsonl'.format(dataset, mode, withcot), 'w') as f:
            for conversations in new_llava_datas:
                f.write(json.dumps(conversations) + "\n")    

if __name__ == '__main__':
    get_new_test_datas()
    # data_paths = {
    #     'twitter15': {
    #         'train': './twitter2015/train.json',
    #         'dev': './twitter2015/dev.json',
    #         'test': './twitter2015/test.json'
    #     },
    #     'twitter17': {
    #         'train': './twitter2017/train.json',
    #         'dev': './twitter2017/dev.json',
    #         'test': './twitter2017/test.json'
    #     },
    # }
    # datasets = ['twitter15', 'twitter17']
    # mode = ['train', 'dev', 'test']
    # withcot = ['no_cot', 'with_cot']
    # answer_format = ['format1', 'format2']
    # for dataset in datasets:
    #     for m in mode:
    #         for cot in withcot:
    #             for af in answer_format:
    #                 get_new_train_datas(data_paths[dataset][m], dataset, m, cot, af)