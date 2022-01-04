The given QA task is divided to two task, multiple choice(mc) and question answering(qa) and I train two models seperately to deal with them.

python3.8 train_mc.py --data_dir ./dataset
python3.8 preprocess.py --data_dir ./dataset
python3.8 train_qa.py --data_dir ./dataset/qa

The multiple choice is stored at 'model/mc', and the question answering model is stored at 'model/qa'. 