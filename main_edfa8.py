from train_model import train
import os

# model name: 'Meta-1' (plus term), 'Meta-2' (scale), 'Meta-3'(shared), NN-DBP
save_path = 'ckpt_set/ckpt_edfa8/'
out_path = 'out_set/out_edfa8/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(out_path):
    os.makedirs(out_path) 
os.system(f'cp config.py {save_path}config.py')
train('NN-DBP',save_path=save_path,out_path=out_path)
train('Meta-1',save_path=save_path,out_path=out_path)
train('Meta-2',save_path=save_path,out_path=out_path)
train('Meta-3',save_path=save_path,out_path=out_path)