import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_plot(df_copy):
    # プロット
    df_plt = df_copy.iloc[2:].copy()
    df_plt.columns = df_plt.iloc[0]
    df_plt = df_plt[1:]
    df_plt = df_plt.iloc[:-3] 
    df_plt.index.name = 'student/item'
    df_for_plot = df_plt.reset_index()
    df_for_plot["C.S.i"] = df_for_plot["C.S.i"].astype(float)
    st.dataframe(df_plt)
    padding = 0.05  # 余白の量を設定
    min_csi = float(df_for_plot["C.S.i"].min())  
    max_csi = float(df_for_plot["C.S.i"].max())  

    # スライダーの値を取得
    threshold_csi = st.slider("Select C.S.i threshold", 0.0, 1.0, 0.5, step=0.05)
    y_line_position = st.slider("Select y-axis line position", 0.0, 99.9, 80.0, step=0.1)
    min_line_position = st.slider("Min line position", min_value=0.0, max_value=y_line_position - 0.1, value=20.0, step=0.1)


    # plotlyを使用して散布図の作成
    fig = px.scatter(df_for_plot, x="C.S.i", y="%", hover_name="student/item", color_discrete_sequence=['black'])

    # 背景色とその他の設定
    fig.update_layout(
        shapes=[
            dict(type="rect", xref="x", yref="y", x0=min_csi, x1=threshold_csi, y0=y_line_position, y1=100, fillcolor="rgba(0, 204, 255, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=threshold_csi, x1=max_csi, y0=y_line_position, y1=100, fillcolor="rgba(0, 204, 204, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=min_csi, x1=threshold_csi, y0=min_line_position, y1=y_line_position, fillcolor="rgba(144, 200, 255, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=threshold_csi, x1=max_csi, y0=min_line_position, y1=y_line_position, fillcolor="rgba(144, 200, 207, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=min_csi, x1=threshold_csi, y0=0, y1=min_line_position, fillcolor="rgba(204, 204, 255, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=threshold_csi, x1=max_csi, y0=0, y1=min_line_position, fillcolor="rgba(204, 204, 204, 0.5)", line=dict(width=0), layer="below"),
        ],
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color="black"),
        xaxis=dict(tickfont=dict(color="black"), titlefont=dict(color="black"), range=[min_csi - padding, max_csi + padding], showgrid=False),
        yaxis=dict(tickfont=dict(color="black"), titlefont=dict(color="black"), showgrid=False)
    )

    # 垂直線と横線を追加
    fig.add_shape(type="line", line=dict(color="black", width=1), x0=threshold_csi, x1=threshold_csi, y0=0, y1=df_for_plot["%"].max())
    fig.add_shape(type="line", line=dict(color="red", width=1), x0=min_csi, x1=max_csi, y0=y_line_position, y1=y_line_position)
    fig.add_shape(type="line", line=dict(color="green", width=1), x0=min_csi, x1=max_csi, y0=min_line_position, y1=min_line_position)

    #疑似凡例
    dummy_x = [0]
    dummy_y = [0]
    fig.update_layout(legend=dict(font=dict(color='black')))
    regions = [
        ('学習安定・良好', 'rgba(0, 204, 255, 0.5)'),
        ('基礎的内容のうっかりミス', 'rgba(0, 204, 204, 0.5)'),
        ('普通', 'rgba(144, 200, 255, 0.5)'),
        ('要注意・レディネス不足', 'rgba(144, 200, 207, 0.5)'),
        ('学力不足', 'rgba(204, 204, 255, 0.5)'),
        ('学習不安定・特異な反応', 'rgba(204, 204, 204, 0.5)'),
    ]

    # 各領域に対応するダミーのトレースを追加
    for desc, color in regions:
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=10, color=color,symbol='square'),
                                name=desc,
                                showlegend=True))

    return fig

def create_cpi_plot(df_copy):
    df_plot_cpi = df_copy.copy()
    df_plot_cpi = df_plot_cpi.iloc[:, :-4]
    st.dataframe(df_plot_cpi)
    df_plot_cpi.loc["C.P.i"] =  df_plot_cpi.loc["C.P.i"].astype(float)
    st.dataframe(df_plot_cpi)
    padding = 0.05
    min_cpi = float(df_plot_cpi.loc["C.P.i"].min())
    max_cpi = float(df_plot_cpi.loc["C.P.i"].max())
    threshold_cpi = st.slider("Select C.P.i threshold", 0.0, 1.0, 0.5, step=0.05)
    #以降の処理のために，df_plot_cpiを変換
    df_for_plot = pd.DataFrame({
        "C.P.i": df_plot_cpi.loc["C.P.i"].values,
        "%": df_plot_cpi.loc["%"].values,
        "Category":df_plot_cpi.loc["カテゴリー"].values,
        "cog":df_plot_cpi.loc["難易度"].values,
        "student/item": df_plot_cpi.loc["生徒/問題"].values
    })
    df_plot_cpi = df_for_plot
    df_plot_cpi['hover_text'] = df_plot_cpi.apply(
        lambda row: (
            f"<span style='color:white;font-weight:bold;'>設問 : {row['student/item']}</span><br><br>"
            f"C.P.i={row['C.P.i']}<br>"
            f"正答率={row['%']}<br>"
            f"単元={row['Category']}<br>"
            f"難易度={row['cog']}"
        ), axis=1
    )


    fig_cpi = px.scatter(
        df_plot_cpi,
        x="C.P.i",
        y="%",
        hover_name="hover_text",  # カスタムホバーテキストを使用
        color_discrete_sequence=['black']
    )

    fig_cpi.update_traces(
        hoverinfo='skip',  # デフォルトのホバー情報を非表示に
        hovertemplate=df_plot_cpi['hover_text']  # カスタムホバーテキストを設定
    )
    fig_cpi.update_layout(
        shapes=[
            dict(type="rect", xref="x", yref="y", x0=min_cpi, x1=threshold_cpi, y0=85, y1=100, fillcolor="rgba(0, 204, 255, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=threshold_cpi, x1=max_cpi, y0=85, y1=100, fillcolor="rgba(0, 204, 204, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=min_cpi, x1=threshold_cpi, y0=15, y1=85, fillcolor="rgba(144, 200, 255, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=threshold_cpi, x1=max_cpi, y0=15, y1=85, fillcolor="rgba(144, 200, 207, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=min_cpi, x1=threshold_cpi, y0=0, y1=15, fillcolor="rgba(204, 204, 255, 0.5)", line=dict(width=0), layer="below"),
            dict(type="rect", xref="x", yref="y", x0=threshold_cpi, x1=max_cpi, y0=0, y1=15, fillcolor="rgba(204, 204, 204, 0.5)", line=dict(width=0), layer="below"),
        ],
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color="black"),
        xaxis=dict(tickfont=dict(color="black"), titlefont=dict(color="black"), range=[min_cpi - padding, max_cpi + padding], showgrid=False),
        yaxis=dict(tickfont=dict(color="black"), titlefont=dict(color="black"), showgrid=False)
    )
    # 垂直線と横線を追加
    fig_cpi.add_shape(type="line", line=dict(color="black", width=1), x0=threshold_cpi, x1=threshold_cpi, y0=0, y1=100)
    fig_cpi.add_shape(type="line", line=dict(color="red", width=1), x0=0, x1=max_cpi, y0=85, y1=85)
    fig_cpi.add_shape(type="line", line=dict(color="green", width=1), x0=0, x1=max_cpi, y0=15, y1=15)
    #疑似凡例
    dummy_x = [0]
    dummy_y = [0]
    fig_cpi.update_layout(legend=dict(font=dict(color='black')))
    regions = [
        ('準良好（やさしすぎ？）', 'rgba(0, 204, 255, 0.5)'),
        ('要検討', 'rgba(0, 204, 204, 0.5)'),
        ('良好', 'rgba(144, 200, 255, 0.5)'),
        ('要検討・不良の可能性', 'rgba(144, 200, 207, 0.5)'),
        ('準良好（難しすぎ？）', 'rgba(204, 204, 255, 0.5)'),
        ('不良', 'rgba(204, 204, 204, 0.5)'),
    ]

    # 各領域に対応するダミーのトレースを追加
    for desc, color in regions:
        fig_cpi.add_trace(go.Scatter(x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=10, color=color,symbol='square'),
                                name=desc,
                                showlegend=True))




    return fig_cpi
