import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Generate some random data
df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

# Build a scatter chart using altair. I modified the example at
# https://altair-viz.github.io/gallery/scatter_tooltips.html
scatter_chart = st.altair_chart(
    alt.Chart(df)
        .mark_circle(size=60)
        .encode(x='a', y='b', color='c')
        .interactive()
)

# Append more random data to the chart using add_rows
for ii in range(0, 100):
    df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    scatter_chart.add_rows(df)
    # Sleep for a moment just for demonstration purposes, so that the new data
    # animates in.
    time.sleep(0.1)

