The preprocess is used to transformed the jsonl data to the json for huggingface api.

python3.8 preprocess.py --data_dir ./data --json_data_dir ./data_json
python3.8 train.py --train_file ./data_json/train.json --validation_file ./data_json/public.json 

To train the model with RL decoder:
python3.8 train_rl.py --train_file ./data_json/train.json --validation_file ./data_json/public.json 