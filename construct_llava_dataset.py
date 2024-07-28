import json
import os
from tqdm import tqdm

def construct_llava_dataset(data_path, dataset, mode, withcot, answer_format):
    with open(data_path, 'r') as f:
        datas = json.load(f)
    
    llava_datas = []

    id_to_sentiment = {
        "POS": "positive",
        "NEG": "negative",
        "NEU": "neutral"
    }
    uid = 1
    for i, data in enumerate(datas):
        if withcot=="no_cot":
            conversion = []
            context = " ".join(data['words'])
            question = "context:{}\nBased on the provided image and context, extract aspect terms and predict sentiment.".format(context)
            conversion.append({
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
            conversion.append({
                "from": "gpt",
                "value": ", ".join(answers)
            })
            llava_datas.append({
                "uid": uid,
                "image": "/nas_mm_2/fangly.fly/data/test/MABSA/{}".format(data['image_id']),
                "conversions": conversion,
                "source": "Twitter-15" if dataset == "twitter15" else "Twitter-17"
            })
            uid += 1
            continue
        elif withcot=="with_cot":
            conversion = []
            context = " ".join(data['words'])
            question = "context:{}\nBased on the provided image and context, extract aspect terms.".format(context)
            conversion.append({
                "from": "human",
                "value": "<image>/nas_mm_2/fangly.fly/data/test/MABSA/{}</image>\n{}".format(data['image_id'],question)
            })
            entitys = []
            for aspect in data['aspects']:
                entitys.append(" ".join(aspect['term']))
            conversion.append({
                "from": "gpt",
                "value": "[" + ", ".join(entitys) + "]"
            })
            llava_datas.append({
                "uid": uid,
                "image": "/nas_mm_2/fangly.fly/data/test/MABSA/{}".format(data['image_id']),
                "conversions": conversion,
                "source": "Twitter-15" if dataset == "twitter15" else "Twitter-17"
            })
            uid += 1
            for aspect in data['aspects']:
                entity = " ".join(aspect['term'])
                conversion = []
                context = " ".join(data['words'])
                question = "context:{}\nBased on the provided image and context, predict the sentiment of {}".format(context, entity)
                conversion.append({
                    "from": "human",
                    "value": "<image>/nas_mm_2/fangly.fly/data/test/MABSA/{}</image>\n{}".format(data['image_id'],question)
                })
                conversion.append({
                    "from": "gpt",
                    "value": "The sentiment of {} is {}".format(entity, id_to_sentiment[aspect['polarity']])
                })
                llava_datas.append({
                    "uid": uid,
                    "image": "/nas_mm_2/fangly.fly/data/test/MABSA/{}".format(data['image_id']),
                    "conversions": conversion,
                    "source": "Twitter-15" if dataset == "twitter15" else "Twitter-17"
                })
                uid += 1
            continue
    if not os.path.exists('./LLava_data/{}'.format(dataset)):
        os.makedirs('./LLava_data/{}'.format(dataset))
    if withcot=="no_cot":
        with open('./LLava_data/{}/{}_{}_{}.jsonl'.format(dataset, mode, withcot, answer_format), 'w') as f:
            for conversion in llava_datas:
                f.write(json.dumps(conversion) + "\n")
    elif withcot=="with_cot":
        with open('./LLava_data/{}/{}_{}.jsonl'.format(dataset, mode, withcot), 'w') as f:
            for conversion in llava_datas:
                f.write(json.dumps(conversion) + "\n")              



if __name__ == '__main__':
    data_paths = {
        'twitter15': {
            'train': './twitter2015/train.json',
            'dev': './twitter2015/dev.json',
            'test': './twitter2015/test.json'
        },
        'twitter17': {
            'train': './twitter2017/train.json',
            'dev': './twitter2017/dev.json',
            'test': './twitter2017/test.json'
        },
    }
    datasets = ['twitter15', 'twitter17']
    mode = ['train', 'dev', 'test']
    withcot = ['no_cot', 'with_cot']
    answer_format = ['format1', 'format2']
    for dataset in datasets:
        for m in mode:
            for cot in withcot:
                for af in answer_format:
                    construct_llava_dataset(data_paths[dataset][m], dataset, m, cot, af)
