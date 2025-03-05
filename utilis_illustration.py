import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def flowCom(speed, agentNum):
    endTrip = 400/speed
    Rnum = 3600/endTrip
    return Rnum*agentNum
def plotFunc(env, vehDataFrame, latLoc, legend=True):
    ##############################################################################
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    prop = fm.FontProperties(family="Times New Roman")#,weight='bold'
    plt.rcParams["font.family"] = "Times New Roman"
    colour = ["r", "g","b" , "c", "m", "y", "tab:purple", "tab:brown",  "k", "pink", "tab:orange", "tab:brown",
              "r", "g","b" , "c", "m", "y", "tab:purple", "tab:brown",  "k", "pink", "tab:orange", "tab:brown",
              "r", "g","b" , "c", "m", "y", "tab:purple", "tab:brown",  "k", "pink", "tab:orange", "tab:brown",
              "r", "g","b" , "c", "m", "y", "tab:purple", "tab:brown",  "k", "pink", "tab:orange", "tab:brown",
              "r", "g","b" , "c", "m", "y", "tab:purple", "tab:brown",  "k", "pink", "tab:orange", "tab:brown",
              "r", "g","b" , "c", "m", "y", "tab:purple", "tab:brown",  "k", "pink", "tab:orange", "tab:brown",
              "r", "g","b" , "c", "m", "y", "tab:purple", "tab:brown",  "k", "pink", "tab:orange", "tab:brown",
              ]
    fig, ax = plt.subplots(1,1, figsize=[10,6])
    for ind, vehNum in enumerate(vehDataFrame.loc["desiredSpeed"].index):
        name = vehNum.split("_")
        name = "agent" + name[1] + f"_{name[2]}"
        t = range(len(env.speedDevScan[vehNum]))
        ax.plot(t, env.speedDevScan[vehNum], color=colour[ind], linewidth=2.0, label=f"{name}")
    # Add some axis labels
    ax.tick_params(axis='both', which='major', labelsize=18)
    #ax.xaxis.set_tick_params(labelsize=20)
    ax.set_xlabel("Time Step (0.25s)", size=22)#fontproperties=prop
    ax.set_ylabel("Speed Deviation (m/s)",fontproperties=prop, size=22)
    # Add a title
    #ax.set_title("Deviation From Desired Speed",size=18)
    ax.grid()
    prop2 = fm.FontProperties(family="Times New Roman",size=14)#,weight='bold'
    #ax.legend(loc='best',prop=prop2)
    #ax.legend(loc='best', bbox_to_anchor=(0.5, .5, 0.5, 0.5), prop=prop2)
    if legend == True:
        ax.legend(loc='center right', bbox_to_anchor=(1.17, .8), prop=prop2)
    fig.show()
    ##############################################################################
    fig1, ax1 = plt.subplots(1,1, figsize=[10,6])
    for ind, vehNum in enumerate(vehDataFrame.loc["desiredSpeed"].index):
        name = vehNum.split("_")
        name = "agent" + name[1] + f"_{name[2]}"
        t = range(len(env.speedDevScan[vehNum]))
        ax1.plot(t, env.jerkScan[vehNum]["jerkLong"], color=colour[ind], linewidth=2.0, label=f"{name}")
    # Add some axis labels
    ax1.tick_params(axis='both', which='major', labelsize=18)
    #ax.xaxis.set_tick_params(labelsize=20)
    ax1.set_xlabel("Time Step (0.25s)", size=22)#fontproperties=prop
    ax1.set_ylabel("Longitudinal Jerk (m/s^3)",fontproperties=prop, size=22)
    # Add a title
    #ax1.set_title("Long. Jerk",size=18)
    ax1.grid()
    prop2 = fm.FontProperties(family="Times New Roman",size=14)#,weight='bold'
    #ax1.legend(loc='best',prop=prop2)
    #ax1.legend(loc='best', bbox_to_anchor=(0.5, .4, 0.3, 0.1), prop=prop2)
    if legend == True:
        ax1.legend(loc='center right', bbox_to_anchor=(1.17, .8), prop=prop2)
    fig1.show()
    ##############################################################################
    fig2, ax2 = plt.subplots(1,1, figsize=[10,6])
    for ind, vehNum in enumerate(vehDataFrame.loc["desiredSpeed"].index):
        name = vehNum.split("_")
        name = "agent" + name[1] + f"_{name[2]}"
        t = range(len(env.speedDevScan[vehNum]))
        ax2.plot(t, env.jerkScan[vehNum]["jerkLat"], color=colour[ind], linewidth=2.0, label=f"{name}")
    # Add some axis labels
    ax2.tick_params(axis='both', which='major', labelsize=18)
    #ax.xaxis.set_tick_params(labelsize=20)
    ax2.set_xlabel("Time Step (0.25s)", size=22)#fontproperties=prop
    ax2.set_ylabel("Lateral Jerk (m/s^2)",fontproperties=prop, size=22)
    # Add a title
    #ax2.set_title("Lat. Jerk",size=18)
    ax2.grid()
    prop2 = fm.FontProperties(family="Times New Roman",size=14)#,weight='bold'
    #ax2.legend(loc='best',prop=prop2)
    #ax2.legend(loc='center', bbox_to_anchor=(0.5, -.04, 0.3, 0.5), prop=prop2)
    if legend == True:
        ax2.legend(loc='center right', bbox_to_anchor=(1.17, .8), prop=prop2)
    fig2.show()
    ##############################################################################
    #latLoc[veh]
    fig3, ax3 = plt.subplots(1,1, figsize=[10,6])
    for ind, vehNum in enumerate(vehDataFrame.loc["desiredSpeed"].index):
        name = vehNum.split("_")
        name = "agent" + name[1] + f"_{name[2]}"
        t = range(len(latLoc[vehNum]))
        ax3.plot(t, latLoc[vehNum], color=colour[ind], linewidth=2.0, label=f"{name}")
    # Add some axis labels
    ax3.tick_params(axis='both', which='major', labelsize=18)
    #ax.xaxis.set_tick_params(labelsize=20)
    ax3.set_xlabel("Time Step (0.25s)", size=22)#fontproperties=prop
    ax3.set_ylabel("Lateral position",fontproperties=prop, size=22)
    # Add a title
    #ax3.set_title("Lat. Jerk",size=18)
    ax3.grid()
    prop2 = fm.FontProperties(family="Times New Roman",size=14)#,weight='bold'
    #ax2.legend(loc='best',prop=prop2)
    if legend == True:
        ax3.legend(loc='center right', bbox_to_anchor=(1.17, .8), prop=prop2)
     ##############################################################################
    """
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = plt.axes(projection ='3d')
    #ax4 = fig4.add_subplot(111, projection='3d')
    for ind, vehNum in enumerate(vehDataFrame.loc["desiredSpeed"].index):
        name = vehNum.split("_")
        name = "agent" + name[1] + f"_{name[2]}"
        t = range(len(latLoc[vehNum]))
        ax4.plot3D(t, longLoc[vehNum], latLoc[vehNum],color=colour[ind], linewidth=2.0, label=f"{name}")
    # Add some axis labels
    ax4.tick_params(axis='both', which='major', labelsize=18)
    #ax.xaxis.set_tick_params(labelsize=20)
    ax4.set_xlabel("Time Step (0.25s)", size=22)#fontproperties=prop
    ax4.set_ylabel("Lateral position",fontproperties=prop, size=22)
    # Add a title
    #ax3.set_title("Lat. Jerk",size=18)
    ax4.grid()
    prop2 = fm.FontProperties(family="Times New Roman",size=14)#,weight='bold'
    #ax2.legend(loc='best',prop=prop2)
    if legend == True:
        ax4.legend(loc='center right', bbox_to_anchor=(1.17, .8), prop=prop2)
   """
    fig3.show()
from mpl_toolkits import mplot3d
def singlePlot(x, y, x_title, y_title, col='red', yspeed=True, marker='*', linestyle='-'):
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    prop = fm.FontProperties(family="Times New Roman")  # ,weight='bold'
    plt.rcParams["font.family"] = "Times New Roman"
    colour = ["r", "g", "b", "c", "m", "y", "tab:purple", "tab:brown", "k", "pink", "tab:orange", "tab:brown"]
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    ax.plot(x, y, color=col, marker='o', linestyle='-', linewidth=2.0)
    # Add some axis labels
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.xaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(x_title, size=22)  # fontproperties=prop
    ax.set_ylabel(y_title, fontproperties=prop, size=22)
    # ax.set_yticks(y)
    # ax.xaxis.set_ticks(range(4000, 20000, 2000))
    # ax.xaxis.set_ticks(np.arange(40, 220, 20))
    if yspeed == True:
        ax.yaxis.set_ticks(np.arange(23.5, 29.5, .5))
    else:
        ax.yaxis.set_ticks(range(4000, 20000, 2000))
    ax.xaxis.set_ticks(range(40, 220, 20))
    # else: ax.xaxis.set_ticks(np.arange(23.5, 29.5, .5))

    # ax.yaxis.set_ticks(range(0, 10, 18000))
    # Add a title
    # ax3.set_title("Lat. Jerk",size=18)
    ax.grid()
    prop2 = fm.FontProperties(family="Times New Roman", size=14)  # ,weight='bold'
    plt.show()

