import json

if __name__ == "__main__":
    data_paths = ["./twitter2015/train.json", "./twitter2015/dev.json", "./twitter2015/test.json", "./twitter2017/train.json", "./twitter2017/dev.json", "./twitter2017/test.json"]

    for data_path in data_paths:
        with open(data_path, "r") as f:
            datas = json.load(f)
        new_datas = []
        for i, data in enumerate(datas, start=1):
            new_data = {
                'question_id': i,
                'image': data['image_id'],
                'text': "Please describe the image.",
            }
            new_datas.append(new_data)
        with open(data_path.replace(".json", "_get_caption.jsonl"), "w") as f:
            for data in new_datas:
                f.write(json.dumps(data) + "\n")