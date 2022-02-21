import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

fig, ax = plt.subplots()

max_x = 50
max_rand = 100
frame_count = 100

x = np.arange(0, max_x)
ax.set_ylim(0, max_rand)

data = np.random.randint(0, max_rand, frame_count+max_x)


line, = ax.plot(x, data[0:max_x])
the_plot = st.pyplot(plt)




def animate(i):  # update the y values (every 1000ms)
    line.set_ydata(data[i:max_x+i])
    the_plot.pyplot(plt)

for i in range(frame_count):
    animate(i)
    time.sleep(0.005)
