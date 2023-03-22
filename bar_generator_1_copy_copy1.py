# !pip install retrying
# !pip install kaleido
# !pip install plotly==5.8

import os
import shutil
import random
import numpy as np
import pandas as pd
import retrying
import random
import string
from random import randint
import random as rand
import tensorflow as tf
from tqdm import tqdm
import plotly.express as px

import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.io._orca

tf.get_logger().setLevel('ERROR')

# Launch orca
unwrapped = plotly.io._orca.request_image_with_retrying.__wrapped__
wrapped = retrying.retry(wait_random_min=1000)(unwrapped)
plotly.io._orca.request_image_with_retrying = wrapped

MAX_Y = 100
MIN_Y = 0


# These set the ordering of bars: Ascending, descending or normal
PLOT_ORDER = {"0": 0, "1": 0, "2": 0}
SIZEREFS = {"1": 7.0, "2": 2.5 , "3": 2.0}

def generate_metadata(min_y = 0, max_y = 100):
    num_bars = 10 # np.random.randint(low=3, high=20+1) # Generate random number for number of bars, min: 3, max: 20
    x = list(range(num_bars))
    
    #num_groups = np.random.randint(low=1, high=2+1)
    num_groups = np.random.randint(low=1, high=1+1)
    
    # Generate random values for bars
    y = np.random.randint(low = min_y, high = max_y + 1, size=(num_groups, num_bars))

    # Sort y plots accordingly
    sort_plots(y) 
    
    return {'x': x, 'y': y}

#function sort_plots takes a list of plots and sorts them based on a specific order defined by the dictionary PLOT_ORDER
def sort_plots(plots):
    for i in range(len(plots)):
        t = plots[i]
        if PLOT_ORDER[str(i)] % 3 == 1:
            t = np.sort(t)
        elif PLOT_ORDER[str(i)] % 3 == 2:
            t = np.sort(t)
            t = np.flip(t)
            
        plots[i] = t
        PLOT_ORDER[str(i)] += 1

        
#function generate_data generates a specified number of samples of data for plotting and returns 3 lists: data, metadata and circle_data        
def generate_data(num_samples = 1):
    data = list()
    metadata = list()
    circle_data = list()
    
    for i in tqdm(range(num_samples), desc="generating metadata: "):
        
        sample = {}
        
        sample_metadata = generate_metadata()
        
        
        sample["y_values"] = sample_metadata["y"]
        sample["x_values"] = sample_metadata["x"]
              
        num_bars = len(sample["y_values"][0])
        num_groups = len(sample["y_values"])
        min_x = np.amin(sample["x_values"])
        max_x = np.amax(sample["x_values"])
        min_y = np.amin(sample["y_values"])
        max_y = np.amax(sample["y_values"])
        
        sample_styles = generate_styles(num_bars=num_bars, num_groups = num_groups, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        
        sample["marker_colors"] = sample_styles["marker_colors"]
        sample["marker_colors_rgba"] = sample_styles["marker_colors_rgba"]
        sample["plot_bg_color"] = sample_styles["plot_bg_color"]
        sample["plot_bg_color_rgba"] = sample_styles["plot_bg_color_rgba"]
        sample["bargap"] = sample_styles["bargap"]
        sample["bargroupgap"] = sample_styles["bargroupgap"]
        sample["x_tick_start"] = sample_styles["xaxis"]["tick0"]
        sample["x_tick_dist"] = sample_styles["xaxis"]["dtick"]
        sample["y_tick_start"] = sample_styles["yaxis"]["tick0"]
        sample["y_tick_dist"] = sample_styles["yaxis"]["dtick"]
        
        circle_data_sample = generate_circle_bars(sample)
        
        data.append(sample)
        metadata.append(sample_metadata)
        circle_data.append(circle_data_sample)
        
    return data, metadata, circle_data

#function generate_marker_colors generates marker colors for bars in a plot
def generate_marker_colors(num_groups, num_bars):
    one_color = np.random.randint(2) # Generate one color or different colors for bars
    
    # Avoid full black or full white
    if num_groups > 1:   #if num_groups > 1, then it generates a list marker_colors of num_groups random colors and a list marker_colors_as_str of their string representations
        marker_colors = [(np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.round(np.random.uniform(0.5, 1), 2)) for i in range(num_groups)]
        marker_colors_as_str = ["rgba" + str(c) for c in marker_colors]  
    elif one_color:     #then it generates a single random color for all bars and its string representation
        marker_colors = (np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.round(np.random.uniform(0.5, 1), 2))
        marker_colors_as_str = "rgba" + str(marker_colors)
    else:      #then it generates a list marker_colors of num_bars random colors and a list marker_colors_as_str of their string representations
        marker_colors = [(np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.round(np.random.uniform(0.5, 1), 2)) for i in range(num_bars)]
        marker_colors_as_str = ["rgba" + str(c) for c in marker_colors]
        
    return marker_colors, marker_colors_as_str

#compute_sizeref computes the sizeref value that will be used to set the size of markers in the plot
def compute_sizeref(plot_size = 500, num_groups = 1, rangey = 100):
    width = (plot_size / rangey) / (num_groups * 2 - 1)
    return ((2.0 *  width)/((3 * width)**2))

def compute_scatter_width(plot_size = 500, num_groups = 1, num_bars = 20):
    div = 4 if num_groups == 2 else 3
    temp_width = (plot_size / (((num_bars - 1) * (num_groups + (num_groups / div))) + num_groups)) / num_groups
    width = 7.5 if temp_width > 7.5 else temp_width
    if width >= 7.5 and num_groups == 2:
        width = 5.5
    return SIZEREFS[str(num_groups)] * (20.0 / num_bars)

#generate_bar_widths generates the widths of the bars in a bar plot
def generate_bar_widths(num_bars):
    one_width = np.random.randint(2) # Generate one width or different widths for bars
    if one_width:
        widths = np.random.uniform(0.1, 1)
    else:
        widths = np.random.uniform(low = 0.1, high = 1, size=num_bars)
    return widths

def generate_bar_gap():
    return np.round(np.random.uniform(0, 0.3), 2)

def generate_bar_group_gap():
    return np.round(np.random.uniform(0, 0.3), 2)

def generate_tick_size(min, max):
    if min <= 0:
        start = 0
    else: 
        start = np.random.randint(0, min)
    
    # Minimum number of ticks is 2, max is 20 
    size = np.round((max - min) / np.random.randint(2, 20 + 1), 2)
    return start, size

def generate_plot_bgcolor():
    is_white = np.random.randint(2) # White or any color
    if is_white:
        bg_color = (255, 255, 255, 1)
    else:
        bg_color = (np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.round(np.random.uniform(0, 0.35), 2))

    bg_color_rgba = "rgba" + str(bg_color)
    
    return bg_color, bg_color_rgba

#generate random values for various plot attributes
def generate_plot_paper_color():
    is_white = np.random.randint(2) # White or any color
    if is_white:
        paper_color = (255, 255, 255, 1)
    else:
        paper_color = (np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.round(np.random.uniform(0, 0.35), 2))

    paper_color_rgba = "rgba" + str(paper_color)
    
    return paper_color, paper_color_rgba

# Genere points for circle bar chart
def generate_circle_bars_one(data):
    num_bars = len(data["x_values"])
    num_groups = len(data["y_values"])
    width  = 30.75 # compute_scatter_width(num_groups = num_groups, num_bars = num_bars)
    group_points = []
    
    for i in range(num_groups):
        bar_points = []
        for j in range(num_bars):
            x = data["x_values"][j] + (data["x_values"][j] * 0.0)
            pointsx = [x]
            pointsy = [0]
            widths = [0]
            endy = data["y_values"][i][j]
            curryst = 4.25
            
            if endy < 2.5:
                continue

            while(curryst <= endy):
                pointsx.append(x)
                pointsy.append(curryst)
                widths.append(width)

                curryst = curryst + 7.5

            bar_points.append({"x": pointsx, "y": pointsy, "widths": widths})

        group_points.append(bar_points)
        
    return group_points

def generate_circle_bars_two(data):
    num_bars = len(data["x_values"])
    num_groups = len(data["y_values"])
    width  = 11.5 # compute_scatter_width(num_groups = num_groups, num_bars = num_bars)
    group_points = []
    
    for i in range(num_groups):
        bar_points = []
        for j in range(num_bars):
            x = data["x_values"][j] + (i * 0.35)
            pointsx = [x]
            pointsy = [0]
            widths = [0]
            endy = data["y_values"][i][j]
            curryst = 2

            while(curryst < endy):
                pointsx.append(x)
                pointsy.append(curryst)
                widths.append(width)

                curryst = curryst + 3

            bar_points.append({"x": pointsx, "y": pointsy, "widths": widths})

        group_points.append(bar_points)
        
    return group_points

def generate_circle_bars_three(data):
    num_bars = len(data["x_values"])
    num_groups = len(data["y_values"])
    width  = 8.125 #compute_scatter_width(num_groups = num_groups, num_bars = num_bars) 
    group_points = []
    
    for i in range(num_groups):
        bar_points = []
        for j in range(num_bars):
            x = data["x_values"][j] + (i * 0.25)
            pointsx = [x]
            pointsy = [0]
            widths = [0]
            endy = data["y_values"][i][j]
            curryst = 1.5

            while(curryst < endy):
                pointsx.append(x)
                pointsy.append(curryst)
                widths.append(width)

                curryst = curryst + 2.125

            bar_points.append({"x": pointsx, "y": pointsy, "widths": widths})

        group_points.append(bar_points)
        
    return group_points

def generate_circle_bars(data):
    num_groups = len(data["y_values"])
    
    if num_groups == 1:
        return generate_circle_bars_one(data)
    if num_groups == 2:
        return generate_circle_bars_two(data)
    if num_groups == 3:
        return generate_circle_bars_three(data)

#generate a dictionary of styles that can be used to customize the appearance of the bar chart
def generate_styles(num_bars, num_groups, min_x, max_x, min_y, max_y):

    (marker_colors, marker_colors_rgba) = generate_marker_colors(num_groups, num_bars)
    (plot_bg_color, plot_bg_color_rgba) = generate_plot_bgcolor()
    bargap = generate_bar_gap()
    bargroupgap = generate_bar_group_gap()
    (xtick_start, xtick_size) = generate_tick_size(min_x, max_x)
    (ytick_start, ytick_size) = generate_tick_size(min_y, max_y)
    
    styles = {
        "marker_colors": marker_colors,
        "marker_colors_rgba": marker_colors_rgba,
        "plot_bg_color": plot_bg_color,
        "plot_bg_color_rgba": plot_bg_color_rgba,
        "bargap": bargap,
        "bargroupgap": bargroupgap,
        "xaxis": {
            "tick0": xtick_start,
            "dtick": xtick_size
        },
        "yaxis": {
            "tick0": ytick_start,
            "dtick": ytick_size
        }
    }
    
    return styles
    
    

#grid_dash=random.choice(open("grid.txt","r").readline().split())
global_grid_probability = 1 #globally defining the grid probability 
image_counter = 0

# Draw source domain
def write_source_data_1(data, filepath, figsize=(600, 512), draw_grid=False, tick_step=10, grid_probability=global_grid_probability):

    global image_counter

    linewidth = random.uniform(1,5)   #to make x  and y axis select the same random number for width between 1 and 5 while generating images
    
    alphabet = string.ascii_lowercase
    word = ''.join(random.choice(alphabet) for i in range(randint(2,3)))    
    
    fig = go.Figure()

    for r in range(len(data["y_values"])):       
        fig.add_trace(go.Bar(y=data["x_values"],    #y parameters specify the vertical positions of the bars
                        x=data["y_values"][r],      #x parameters specify the horizontal positions of the bars
                        orientation="h",  #generates horizontal plots
                        width=0.7,   #sets the width of bar
                        #orientation=random.choice(open("orientation.txt","r").readline().split()),
                        name=word,       #name parameter sets the label for each group of bars, and it refers to word which generates values randomly
                        #name=random.choice(open("legend.txt","r").readline().split()),
                        marker_color=data["marker_colors_rgba"][r] if len(data["y_values"]) > 1 else data["marker_colors_rgba"],
                        marker_line_width=2   #set the width of the border around the bars
                        ))
    size=random.randint(19,23)  #randomly sets the size of x and y axis titles between 19 to 23
                                         
    fig.update_layout(
        margin=dict(l=5, r=25, t=30, b=25),
        plot_bgcolor=data["plot_bg_color_rgba"],
        width=figsize[0], height=figsize[1],
        #title=random.choice(open("randomFile.txt","r").readline().split()),
        xaxis_range=[0,110],
        yaxis_range=[-0.5,10],
        xaxis_title=random.choice(open("label.txt","r").readline().split()),   #title of x axis is randomly selected from file label.txt
        yaxis_title=random.choice(open("label.txt","r").readline().split()),   #title of y axis is randomly selected from file label.txt
        xaxis_title_font=dict(size=size),
        yaxis_title_font=dict(size=size),
        #bargap=data["bargap"],
        bargroupgap=data["bargroupgap"],
        showlegend=True,
        #axis_width=randint(1,5),
        xaxis={
                "showline": True, 
                #"linewidth": random.uniform(1,5), 
                "linewidth": linewidth,
                "linecolor": 'black',
                "dtick": tick_step
            },
            
        yaxis={
                "showline": True, 
                #"linewidth": random.uniform(1,5), 
                "linewidth": linewidth,
                "linecolor": 'black',
                "ticks": "outside",
                #"dtick": tick_step
            })

    fig.update_layout(bargap=0.11)
    #fig.add_annotation(ax=-2.5, axref='x', ay=0, ayref='y', x=9.8, xref='x', y=0, yref='y', arrowwidth=random.randint(2,5), arrowhead=2)
    #fig.add_annotation(ax=15, axref='x', ay=0, ayref='y', x=0, xref='x', y=0, yref='y', arrowwidth=random.randint(2,5), arrowhead=2)
    
    with open("x.txt","r") as f:        #x.txt has values for x anchor that decide the positioning of the legend
    	x_options = f.readline().split()
    
    with open("y.txt","r") as f:        #y.txt has values for y anchor that decide the positioning of the legend
    	y_options = f.readline().split()
                   	
    fig.update_layout(legend=dict(
    	orientation=random.choice(open("orientation.txt","r").readline().split()),
    	
    	font_family=random.choice(open("font.txt","r").readline().split()),  #randomly selects a font for every individual image from font.txt file
    	x=float(random.choice(x_options)),
    	y=float(random.choice(y_options))
    	#yanchor=random.choice(open("yanchor.txt","r").readline().split()),
    	#xanchor=random.choice(open("xanchor.txt","r").readline().split())
    ))
      
    alphabet = string.ascii_lowercase   #picks a random string of alphabets from a to z


    ticktext = []
    for i in range(10):      
    	word = ''.join(random.choice(alphabet) for i in range(3))  #uses string of alphabets picked above as the ticktext for graphs
    	ticktext.append(word)     
          
    fig.update_layout(
    	yaxis = dict(
    	tickmode = 'array',
    	#tickvals = [20,40,60,80,100,120,140,160,180,200],
    	tickvals = [0.1,1,2,3,4,5,6,7,8,9],   #decides tick positioning 
        ticktext = ticktext
    ))  
    
    #df = px.data.tips()     
    #fig = px.bar(
    	#df,
    	#orientation='h',
    	#)        
    
    
    fig.update_layout(
    	font=dict(
    		size=random.randint(13,22),    #randomly picks a number between 13 to 22 for font size of the graphs
    		family=random.choice(open("font.txt","r").readline().split())
    		)
    	)
    	
    fig.update_layout(
    	title_text=random.choice(open("randomFile.txt","r").readline().split()),  #generates title text by randomly selecting words from randomFile.txt
    	title_x=random.uniform(0.0,0.9),   #positioning of the chart title
    	title_font_size=random.randint(25,35),   #size of the title 
    	margin={'t':50, 'b':50, 'l':1}, #for making the whole title is visible  
    	title_font_family=random.choice(open("font.txt","r").readline().split())  #font style for the chart title
    		) 
    
    		    	
        
    #plt.title(random.choice(open("randomFile.txt","r").readline().split()), fontsize = random.randint(11,18))
    #plt.xlabel(random.choice(open("myFile.txt","r").readline().split()), fontsize = random.randint(10,15))
    #plt.ylabel(random.choice(open("myFile.txt","r").readline().split()), fontsize = random.randint(10,15))
    
    if (image_counter % 5 == 4):   #every 4th image, per 5 images should not have a grid(this is to produce 80% of images having grids)
    	draw_grid = False
    else:
    	draw_grid = random.random() < grid_probability
    	
# Update the x-axes with or without gridlines
    fig.update_xaxes(showgrid=False)
    if draw_grid:
        fig.update_xaxes(showgrid=True, gridcolor='#aaaaaa', gridwidth=1, griddash=random.choice(open("grid.txt","r").readline().split()))


# Write the image to file
    pio.write_image(fig=fig, file=filepath, format="png", width=figsize[0], height=figsize[1])

    image_counter += 1      

#global_grid_probability = 1
image_counter = 0  #resetting the image counter
 
#Draw target domain for channelwise model  
def write_circle_target_data_1(data, filepath, figsize=(600, 512), draw_grid=False, tick_step=10, grid_probability=global_grid_probability):

    global image_counter
    fig = go.Figure()
    for i in range(len(data)):
        for j in range(len(data[i])):                              
            n = 1
              
            fig.add_trace(go.Scatter(x=data[i][j]["y"][::n],
                        y=data[i][j]["x"][::n],
                        marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[0]/figsize[1]),        
            "sizemode":'diameter',
            "sizeref": 0.82,     #decides size of the circles
                                 },
                        )) 

    fig.update_traces(mode='markers', marker_line_width=1, visible=False,)

    fp_parts = filepath.rsplit(".", 1)
    fig.update_layout(
        plot_bgcolor="white",   
        margin=dict(l=30, r=30, t=30, b=30),
        width=figsize[0], height=figsize[1],
        xaxis_range=[0,110],
        yaxis_range=[-0.5,10],
        #orientation="h",
        #barmode="relative",
        yaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black'},
        xaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black', "ticks": "outside", "dtick": tick_step, "ticklen": 10, "tickwidth": 2},
        showlegend=False
    )   
    
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_axes.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])  #generates axes portion of target
    
    draw_grid = random.random() < grid_probability and image_counter % 5 != 0   #4th image per 5 images will not have grid
    
    # Update the y-axes with or without gridlines
    fig.update_xaxes(showgrid=False)
    if draw_grid:
        fig.update_xaxes(showgrid=True, gridcolor='black', griddash='dash', gridwidth=1)

    fig.update_layout(yaxis={"linecolor": 'white'}, xaxis={"linecolor": 'white', "ticklen": 0, "tickwidth": 0})
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_grids.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])  #generates grids portion of target

    fig.update_traces(mode='markers', marker_line_width=2, marker_color="white", marker_line_color="black", visible=True)
    fig.update_layout(xaxis={"showgrid":False})
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_content.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])   #generates content portion of target
    
    image_counter += 1
    
#global_grid_probability = 1
image_counter = 0   #resetting the image counter
#Draw target domain for RGB Model    
def write_circle_target_data1_1(data, filepath, figsize=(600, 512), draw_grid=False, tick_step=10, grid_probability=global_grid_probability):

    global image_counter
    fig = go.Figure()
    for i in range(len(data)):
        for j in range(len(data[i])):
            n = 1
            
            fig.add_trace(go.Scatter(x=data[i][j]["y"][::n],
                        y=data[i][j]["x"][::n],
                        marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[0]/figsize[1]),    
                        #marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[1]/figsize[0]),     
            "sizemode":'diameter',
            "sizeref": 0.82,     #decides size of the circles
                                 },
                        ))
                        
    
    fig.update_traces(mode='markers', marker_line_width=1, visible=False)

    fp_parts = filepath.rsplit(".", 1)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=30, b=30),
        width=figsize[0], height=figsize[1],
        xaxis_range=[0,110],
        yaxis_range=[-0.5,10],
        yaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black'},
        xaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black', "ticks": "outside", "dtick": tick_step, "ticklen": 0, "tickwidth": 0, "ticklen": 10, "tickwidth": 2, "showgrid":False},
        showlegend=False
    )  
    
    # Always set draw_grid to False on the 4th, 9th, 14th, 19th, and so on images
    draw_grid = random.random() < grid_probability and image_counter % 5 != 1

    fig.update_traces(mode='markers', marker_line_width=2, marker_color="white", marker_line_color="black", visible=True)
    if draw_grid:
    	fig.update_xaxes(showgrid=True, gridcolor='black', griddash='dot', gridwidth=1)
    pio.write_image(fig=fig, file=f"{fp_parts[0]}.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])
    image_counter += 1    


# Convert data to numpy dataframe
def serialize_data(data, headers):
    df = {}
    
    for h in headers:
        df[h] = []
    
    for d in data:
        for h in d: 
            val_str = ""
            if h == "x":
                val_str +=  ", ".join(map(str, d[h]))
                val_str += " ."
            else:
                for vals in d[h]:
                    val_str +=  ", ".join(map(str, vals))
                    val_str += " ."
            df[h].append(val_str)
            
    return pd.DataFrame(df)
    
    

