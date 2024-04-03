#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import sys
import matplotlib

suffixes=["_p2p", "_p2p_seq_IO", "_p2p_par_IO" ,"_rma", "_rma_seq_IO", "_rma_par_IO"] 

# adjust the plot settings for your screen size (or resize your browser window when displaying the plots using the HTML file)
plot_width  = 8 
plot_height = 6
legend_font_size = 8
plot_filetype = ".svg"
dpi = None
legend_box_left = 0.45
legend_box_top = 1.14

# set the font size for the plots
SMALL_SIZE = 11
MEDIUM_SIZE = 13
LARGE_SIZE = 15

matplotlib.rc('font', size=SMALL_SIZE)        
matplotlib.rc('axes', titlesize=SMALL_SIZE)   
matplotlib.rc('axes', labelsize=SMALL_SIZE)  
matplotlib.rc('xtick', labelsize=SMALL_SIZE)  
matplotlib.rc('ytick', labelsize=SMALL_SIZE)  
matplotlib.rc('legend', fontsize=SMALL_SIZE)  
matplotlib.rc('figure', titlesize=LARGE_SIZE) 

# check if "png" is passed as the 1st argument and change the plot settings
if len(sys.argv) > 1 and sys.argv[1] == "png":
    plot_filetype = ".png"
    dpi = 400

x_axis_points = [1, 16, 32, 64, 128]
#x_axis_points = [1,2,4]
data = []
with open('run_full_mpi_2d_out.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    data = list(csvreader)
    
data_dict = {}
headerlen = 1

for i in range(0,len(data)-headerlen):
    suffix = suffixes[i % 6]
    row = data[i + headerlen]
    datakey = str(row[4]) + suffix
    iterationtime = float(row[13])
    if not datakey in data_dict.keys():
        data_dict[datakey] = []
    data_dict[datakey].append(iterationtime)

for comm_type in ['p2p', 'rma']:

    # scaling
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            ax.plot(x_axis_points[0:len(scaling_values)], scaling_values, marker=plot_marker, label=key)

    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(scaling_values)])
    ax.set_ylabel('Iteration time [ms]')
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_scaling_mpi_" + comm_type + plot_filetype, dpi=dpi)

    # speedup
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]

            plt.plot(x_axis_points[0:len(speedup_values)], speedup_values, marker=plot_marker, label=key)

    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(speedup_values)])
    ax.set_ylabel('Speedup')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_speedup_mpi_" + comm_type + plot_filetype, dpi=dpi)

    # efficiency
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]
            efficiency_values = []
            for i in range(0,len(speedup_values)):
                efficiency_values.append(speedup_values[i]/x_axis_points[i]*100)

            plt.plot(x_axis_points[0:len(efficiency_values)], efficiency_values, marker=plot_marker, label=key)

            
    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(efficiency_values)])
    ax.set_ylabel('Efficiency [%]')
    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.grid(True)
#plt.show()
    plt.tight_layout()
    plt.savefig("ppp_efficiency_mpi_" + comm_type + plot_filetype, dpi=dpi)



x_axis_points = [1, 4*8, 16*8, 32*8]
#x_axis_points = [1,2,4]
data = []
with open('run_full_hybrid_2d_out.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    data = list(csvreader)


data_dict = {}


for i in range(0,len(data)-headerlen):
    suffix = suffixes[i % 6]
    row = data[i + headerlen]
    datakey = str(row[4]) + suffix
    iterationtime = float(row[13])
    if not datakey in data_dict.keys():
        data_dict[datakey] = []
        data_dict[datakey].append(iterationtime)
    else:
        data_dict[datakey].append(iterationtime)


for comm_type in ['p2p', 'rma']:

    # scaling
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            plt.plot(x_axis_points[0:len(scaling_values)], scaling_values, marker=plot_marker, label=key)

    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(scaling_values)])
    ax.set_ylabel('Iteration time [ms]')
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_scaling_hybrid_" + comm_type + plot_filetype, dpi=dpi)

    # speedup
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]

            plt.plot(x_axis_points[0:len(speedup_values)], speedup_values, marker=plot_marker, label=key)

    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(speedup_values)])
    ax.set_ylabel('Speedup')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_speedup_hybrid_" + comm_type + plot_filetype, dpi=dpi)

    # efficiency
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]
            efficiency_values = []
            for i in range(0,len(speedup_values)):
                efficiency_values.append(speedup_values[i]/x_axis_points[i]*100)

            plt.plot(x_axis_points[0:len(efficiency_values)], efficiency_values, marker=plot_marker, label=key)

            
    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(efficiency_values)])
    ax.set_ylabel('Efficiency [%]')
    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_efficiency_hybrid_" + comm_type + plot_filetype, dpi=dpi)


x_axis_points = [1, 2*16, 4*16, 8*16, 16*16]
#x_axis_points = [1,2,4]
data = []
with open('run_full_hybrid_1d_out.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    data = list(csvreader)


data_dict = {}

for i in range(0,len(data)-headerlen):
    suffix = suffixes[i % 6]
    row = data[i + headerlen]
    datakey = str(row[4]) + suffix
    iterationtime = float(row[13])
    if not datakey in data_dict.keys():
        data_dict[datakey] = []
        data_dict[datakey].append(iterationtime)
    else:
        data_dict[datakey].append(iterationtime)

for comm_type in ['p2p', 'rma']:

    # scaling
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            plt.plot(x_axis_points[0:len(scaling_values)], scaling_values, marker=plot_marker, label=key)

    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(scaling_values)])
    ax.set_ylabel('Iteration time [ms]')
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_scaling_hybrid_1D_" + comm_type + plot_filetype, dpi=dpi)

    # speedup
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]

            plt.plot(x_axis_points[0:len(speedup_values)], speedup_values, marker=plot_marker, label=key)

    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(speedup_values)])
    ax.set_ylabel('Speedup')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_speedup_hybrid_1D_" + comm_type + plot_filetype, dpi=dpi)

    # efficiency
    fig, ax = plt.subplots(figsize=[plot_width, plot_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]
            efficiency_values = []
            for i in range(0,len(speedup_values)):
                efficiency_values.append(speedup_values[i]/x_axis_points[i]*100)

            plt.plot(x_axis_points[0:len(efficiency_values)], efficiency_values, marker=plot_marker, label=key)

            
    ax.set_xlabel('Number of cores')
    ax.set_xticks(x_axis_points[0:len(efficiency_values)])
    ax.set_ylabel('Efficiency [%]')
    ax.legend(loc='upper center', bbox_to_anchor=(legend_box_left, legend_box_top), ncol=5, fancybox=True, prop={'size': legend_font_size})
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("ppp_efficiency_hybrid_1D_" + comm_type + plot_filetype, dpi=dpi)