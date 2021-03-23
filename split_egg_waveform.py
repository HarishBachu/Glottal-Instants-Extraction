import numpy as np
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm, trange 



def split_speech_signal(home_dir, desc):

    speech_dir = os.path.join(home_dir, "Speech")
    egg_dir = os.path.join(home_dir, "EGG")
    os.mkdir(speech_dir)
    os.mkdir(egg_dir)
    
    for i in tqdm(os.listdir(home_dir), desc = desc, ncols = 80):
        path_to_wav = os.path.join(home_dir, i)
        dest_path_speech = os.path.join(speech_dir, i)
        dest_path_egg = os.path.join(egg_dir, i)

        os.system('ch_wave -c 0 -F 16000 {0} -o {1}'.format(path_to_wav, dest_path_speech))
        os.system('ch_wave -c 1 -F 16000 {0} -o {1}'.format(path_to_wav, dest_path_egg))

def main():

    home_dir = [
        '/home/harish/Documents/cmu_us_bdl_arctic-WAVEGG/cmu_us_bdl_arctic/orig/', 
        '/home/harish/Documents/cmu_us_jmk_arctic-WAVEGG/cmu_us_jmk_arctic/orig/', 
        '/home/harish/Documents/cmu_us_slt_arctic-WAVEGG/cmu_us_slt_arctic/orig/'
    ]
    desc = ["Extracting BDL", "Extracting JMK", "Extracting SLT"]
    # home_dir = input("Enter Root Directory: ")
    # print()
    for i in range(3):
        split_speech_signal(home_dir = home_dir[i], desc = desc[i])        

if __name__ == "__main__":
    main()

    