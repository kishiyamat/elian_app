from io import StringIO
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


def minmax_scale(x):
    res = (x - np.min(x)) / (np.max(x) - np.min(x))  # +0.5
    return res  # + 0.5


def from_zero(x):
    res = x - min(x)  # ( - np.mean(x)) / np.std(x, ddof=1)
    return res


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

    @staticmethod
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

    @property
    def df(self):
        data = self.data
        stroke_id = 0
        stroke, strokes, = (
            [],
            [],
        )

        for xytz in data:
            if xytz != self.stroke_boundary:
                sxytz = [stroke_id] + list(np.fromstring(xytz, dtype=int, sep=" "))
                stroke.append(np.array(sxytz, dtype=int))
            else:  # "0 0 0"
                if self.stroke_validation(stroke):
                    strokes.append(np.array(stroke))
                    stroke_id += 1
                stroke = []

        x_y_ms_z = np.concatenate(strokes)
        x_y_ms_z_df = pd.DataFrame(x_y_ms_z, columns=["stroke", "x", "y", "ms", "z"])

        df = x_y_ms_z_df
        pipe = Pipeline([("cluster", KMeans(n_clusters=self.n_aoi, random_state=42))])
        df["cluster"] = pipe.fit_predict(df[["x", "y"]])
        return df

st.header("Elian Visualizer")
st.subheader("How to use")
st.write("""
1. Drag and drop files here から `.elian` ファイルをアップロード(同時に複数可能)
1. 見栄えを調整(左の Figure size や Font size を調整してください。)
1. 分割数を調整(左の Number of ... から調整してください。)
1. もし時間情報が不要なら、Alpha (透明度) for time のチェックをオフ

(初回起動時はちょっともたつきます。)
""")

# ハイパラ
figure_size = st.sidebar.slider("Figure size", 5, 10, 8)
font_size = st.sidebar.slider("Font size", 12, 32, 24)
n_components = st.sidebar.slider("Number of pitcure in a file", 1, 5, 4)
use_alpha = st.sidebar.checkbox("Alpha for time", True)

st.subheader("Upload")

# filesを取得
uploaded_files = st.file_uploader("Choose elian files", accept_multiple_files=True)
if len(uploaded_files) == 0:
    raise ValueError("Please upload some elian files!")

st.subheader("Results")
st.write("画像の右に拡大マークがでます。あるいは左のフォントサイズで調整できます。")

for uploaded_file in uploaded_files:
    string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).readlines()
    st.write(f"filename: {uploaded_file.name}")
    df = Elian(string_data, n_aoi=n_components).df

    cols = st.columns(n_components)
    for i in range(n_components):
        with cols[i]:
            df_i = df.query(f"cluster == {i}")
            df_i["stroke"] = from_zero(df_i["stroke"])
            df_i["ms"] = df_i.groupby("stroke").transform(minmax_scale)["ms"]

            plt.figure(figsize=(figure_size, figure_size))
            X = list(df_i["x"])
            Y = list(df_i["y"] * -1)  # 上下逆なため
            S = list(df_i["stroke"])
            T = list(df_i["ms"] * -1 + 1)  # 直感的に後を薄くする

            plt.xlim(min(X) - 5, max(X) + 5)
            plt.ylim(min(Y) - 5, max(Y) + 5)

            for i in range(len(df_i)):
                try:
                    if use_alpha:
                        plt.text(
                            X[i],
                            Y[i],
                            str(S[i]),
                            color=f"C{S[i]}",
                            alpha=T[i],
                            fontsize=font_size,
                        )  # 67はない
                    else:
                        plt.text(
                            X[i], Y[i], str(S[i]), color=f"C{S[i]}", fontsize=font_size
                        )  # 67はない
                except:
                    print("ERROR!")

            st.pyplot(plt)

st.subheader("Appendix")
st.write("""
オンライン版はそこまでパワフルではないので、手元のPCでも走らせられます。
が、ちょっと複雑なので今度にします。
もし他にご要望があれば Slack かメール( kishiyama.t@gmail.com ) で岸山にご連絡ください。
""")