from io import StringIO
import streamlit as st
from typing import List
import pandas as pd
import numpy as np

# for animation in the notebook
from matplotlib import rc
rc('animation', html='html5')

class Elian:
    data_boundary = "[DATA]"
    stroke_boundary = "0 0 0"

    def __init__(self, row_data: List[str]):
        self.row_data = row_data

    @property
    def header(self):
        content: List[str] = [x.strip() for x in self.row_data]
        begins_at: List[str] = content.index(self.data_boundary)
        return content[:begins_at]

    @property
    def data(self):
        """
        :return:
            "[DATA]" から始まる部分を抽出
            '0 0 0' の分割はこと後のステップで実施
        """
        content: List[str] = [x.strip() for x in self.row_data]
        begins_at: List[str] = content.index(self.data_boundary) + 1
        return content[begins_at:]

    @property
    def hoge(self):
        pass

# filesを取得
uploaded_files = st.file_uploader("Choose elian files", accept_multiple_files=True)
uploaded_file = uploaded_files[0]
string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).readlines()
st.write("filename:", uploaded_file.name)
# st.write(Elian(string_data).data)
# st.write(Elian(string_data).header)

# df化
import numpy as np
import pandas as pd
data = Elian(string_data).data
data = list(filter(lambda r: r != "0 0 0", data))
x_y_ms_z = np.array([np.fromstring(r, dtype=int, sep=' ') for r in data])
x_y_ms_z_df = pd.DataFrame(x_y_ms_z, columns=["x","y","ms","z"])

# https://plotnine.readthedocs.io/en/stable/generated/plotnine.animation.PlotnineAnimation.html
# Parameters used to control the spiral
def plot(df, i):
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    z_min, z_max = df["z"].min(), df["z"].max()*2
    p = (ggplot(df.head(i))
         + geom_point(aes('x', 'y', color='z'), size=1)
         + lims(
             x=(x_min, x_max),
             y=(y_min, y_max),
             color=(z_min, z_max)
         )
         + theme_void()
         + theme(
             aspect_ratio=1,
             # Make room on the right for the legend
             subplots_adjust={'right': 0.85}
         )
         + theme(figure_size=(3, 3)) 
    )
    return p


# It is better to use a generator instead of a list
sample_i = 500
tmp = x_y_ms_z_df.head(sample_i)
# generator を使ってみる
plots = map(lambda i: plot(tmp, i), range(1, len(x_y_ms_z_df.head(sample_i))))
ani = PlotnineAnimation(plots, interval=30, repeat=False)
ani.save('animation.mp4')

# https://docs.streamlit.io/library/api-reference/media/st.video
video_file = open('animation.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)