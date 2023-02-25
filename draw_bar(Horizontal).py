import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from bar_generator import generate_data, write_source_data, write_source_data_elongated, write_circle_target_data, write_circle_target_data1, write_circle_target_data_elongated, write_circle_target_data1_elongated, write_source_data_1, write_source_data_2, write_source_data_3, write_circle_target_data_1, write_circle_target_data1_1, write_circle_target_data_2, write_circle_target_data1_2, write_circle_target_data_3, write_circle_target_data1_3, serialize_data

#from bar_generator import generate_data, write_source_data, write_circle_target_data, serialize_data
from utils import postprocessing, maskgen

NUM_SAMPLES = 1000
P_grid = 0.4

os.makedirs("./data/source", exist_ok=True)
os.makedirs("./data/tactile", exist_ok=True)
#os.makedirs("./data/tactile1", exist_ok=True)


with tf.device('/device:GPU:0'):
    data, metadata, circle_data = generate_data(num_samples=NUM_SAMPLES)
    
    

    for i in tqdm(range(len(data)), desc='bar charts: '):

        fig_size = [512,512]
        fig_size_elongated = random.choices([[512,1024], [512,1024], [512,1024]], weights=[.5, .25, .25])[0]
        fig_size_1 = random.choices([[700,622], [700,622], [700,622]], weights=[.25, .5, .5])[0]
        fig_size_2 = random.choices([[512,672], [512,672], [512,672]], weights=[.5, .25, .25])[0]
        fig_size_3 = random.choices([[750,650], [750,650], [750,650]], weights=[.5, .25, .25])[0]
        draw_grid = random.random() < P_grid
        tick_step = random.randint(10, 16)

        write_source_data(data[i], f"./data/source/s_{i+1}.png", fig_size, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+1}.png")
        
        write_circle_target_data(circle_data[i], f"./data/tactile/t_{i+1}.png", fig_size, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+1}.png")
        
        write_circle_target_data1(circle_data[i], f"./data/tactile/t_{i+1}.png", fig_size, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+1}.png")
        
        write_source_data_elongated(data[i], f"./data/source/s_{i+1001}.png", fig_size_elongated, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+1001}.png")
        
        write_circle_target_data_elongated(circle_data[i], f"./data/tactile/t_{i+1001}.png", fig_size_elongated, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+1001}.png")
        
        write_circle_target_data1_elongated(circle_data[i], f"./data/tactile/t_{i+1001}.png", fig_size_elongated, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+1001}.png")
        
        write_source_data_1(data[i], f"./data/source/s_{i+3001}.png", fig_size_1, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+3001}.png")
        
        write_circle_target_data_1(circle_data[i], f"./data/tactile/t_{i+3001}.png", fig_size_1, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+3001}.png")
        
        write_circle_target_data1_1(circle_data[i], f"./data/tactile/t_{i+3001}.png", fig_size_1, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+3001}.png")
        
        write_source_data_2(data[i], f"./data/source/s_{i+4001}.png", fig_size_2, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+4001}.png")
        
        write_circle_target_data_2(circle_data[i], f"./data/tactile/t_{i+4001}.png", fig_size_2, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+4001}.png")
        
        write_circle_target_data1_2(circle_data[i], f"./data/tactile/t_{i+4001}.png", fig_size_2, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+4001}.png")
        
        write_source_data_3(data[i], f"./data/source/s_{i+2001}.png", fig_size_3, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+2001}.png")
        
        write_circle_target_data_3(circle_data[i], f"./data/tactile/t_{i+2001}.png", fig_size_3, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+2001}.png")
        
        write_circle_target_data1_3(circle_data[i], f"./data/tactile/t_{i+2001}.png", fig_size_3, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+2001}.png")
        
        
        #write_circle_target_data2(circle_data[i], f"./data/tactile1/t_{i+5001}.png", fig_size1, draw_grid, tick_step)
        #print(f"./data/tactile1/t_{i+5001}.png")
        
    # metadata_df = serialize_data(metadata, ["x", "y"])
    # metadata_df.to_csv("metadata.csv", index=False)

# !zip -qq -r ./bardata.zip ./data/ ./metadata.csv
