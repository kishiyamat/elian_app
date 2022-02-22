from io import StringIO
from more_itertools import locate

import streamlit as st
from typing import List
import pandas as pd
import numpy as np

# for animation in the notebook
from matplotlib import rc

rc("animation", html="html5")

class Elian:
    data_boundary = "[DATA]"
    stroke_boundary = "0 0 0"

    def __init__(self, row_data: List[str], n_aoi):
        self.row_data = row_data
        self.n_aoi = n_aoi

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
            '0 0 0' の分割でstrokeを算出
        """
        content: List[str] = [x.strip() for x in self.row_data]
        begins_at: List[str] = content.index(self.data_boundary) + 1
        data = content[begins_at:]
        return data

    @property
    def hoge(self):
        pass


# filesを取得
uploaded_files = st.file_uploader("Choose elian files", accept_multiple_files=True)
if len(uploaded_files):
    uploaded_file = uploaded_files[0]
    string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).readlines()
    st.write("filename:", uploaded_file.name)

    # df化
    import numpy as np
    import pandas as pd

    data = Elian(string_data, n_aoi=4).data


    def stroke_validation(stroke: list):
        """_summary_

        Args:
            stroke (list):
                [[stroke_i x y time z(pressure)],
                [stroke_i x y time z(pressure)],
                ...]
        Returns:
            _type_: bool
        """
        if len(stroke) < 2:
            return False
        stroke = np.array(stroke)
        x_diff = np.sum(np.abs(np.diff(stroke[:, 1])))
        y_diff = np.sum(np.abs(np.diff(stroke[:, 2])))
        z_mean = np.mean(stroke[:, 4])
        if x_diff > 1 and y_diff > 1 and z_mean > 1:
            return True
        return False


    strokes = []
    stroke = []
    stroke_id = 0



    for xytz in data:
        if xytz != "0 0 0":
            sxytz = [stroke_id] + list(np.fromstring(xytz, dtype=int, sep=" "))
            stroke.append(np.array(sxytz, dtype=int))
        else:  # "0 0 0"
            if stroke_validation(stroke):
                strokes.append(np.array(stroke))
                # 有効なら更新
                stroke_id += 1
            stroke = []

    # strokes
    boundaries: List[int] = list(locate(data, lambda x: x == Elian.stroke_boundary))
    boundary_starts, boundary_ends = [0] + boundaries, boundaries + [-2]

    # data = list(filter(lambda r: r != "0 0 0", data))
    # x_y_ms_z = np.array([np.fromstring(r, dtype=int, sep=" ") for r in data])
    x_y_ms_z = np.concatenate(strokes)
    x_y_ms_z_df = pd.DataFrame(x_y_ms_z, columns=["stroke", "x", "y", "ms", "z"])

    # time to z
    # cluster by shape
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df = x_y_ms_z_df
    n_components = st.slider('Number of Pitcure', 1, 5, 4)
    pipe = Pipeline([("cluster", KMeans(n_clusters=n_components))])
    df["cluster"] = pipe.fit_predict(df[["x", "y"]])


    def minmax_scale(x):
        res = (x-np.min(x))/(np.max(x)-np.min(x))# +0.5
        return res # + 0.5

    def from_zero(x):
        res = x - min(x)# ( - np.mean(x)) / np.std(x, ddof=1)
        return res

    import matplotlib.pyplot as plt

    for i in range(n_components):
        st.write(f"cluster == {i}")
        # pair_of_strokes = self.df_at_adult(aoi)
        # pair_of_strokes = self.df_at_child(aoi)
        # dataをすでに作成して、Kmeans で分割してみる。
        df_i = df.query(f"cluster == {i}") 
        df_i["stroke"] = from_zero(df_i["stroke"])
        df_i["ms"] = df_i.groupby("stroke").transform(minmax_scale)["ms"]
        pair_of_strokes = df_i

        plt.figure(figsize=(7, 7))
        X = list(pair_of_strokes["x"])
        Y = list(pair_of_strokes["y"] * -1)  # 上下逆なため
        S = list(pair_of_strokes['stroke'])
        T = list(pair_of_strokes['ms'] * -1 + 1)  # 直感的に後を薄くする
        # st.write(T)

        plt.xlim(min(X) - 5, max(X) + 5)
        plt.ylim(min(Y) - 5, max(Y) + 5)

        for i in range(len(pair_of_strokes)):
            try:
                plt.text(X[i], Y[i], str(S[i]), color=f"C{S[i]}", alpha=T[i])  # 67はない
            except:
                print("ERROR!")

        st.pyplot(plt)
