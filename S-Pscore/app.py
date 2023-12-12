import streamlit as st
import pandas as pd
import os
import webbrowser
from sp_function import (preprocess_dataframe,format_and_sort_dataframe,set_column_totals,finalize_dataframe,
                         stock_and_remove,compute_scores_and_ratios,compute_csi,compute_cpi,insert_and_adjust_data,
                         calculate_difference_coefficient,generate_html_table_with_css
                        )
from plot_section import create_plot,create_cpi_plot


def main():
    st.title("S-P Score")

 # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv"])

    if uploaded_file is not None:
            df = pd.read_csv(uploaded_file,header=None)
            first_cell_value = df.iloc[0, 0]
            if first_cell_value == 'S-Pscore':
                
                st.write("S-Pscoreの処理を行います.")
                # CSVファイルをDataFrameとして読み込んで整形
                df, df_shape_col = preprocess_dataframe(df)
                df, num_rows, num_cols = format_and_sort_dataframe(df)
                df, df_Category = set_column_totals(df, num_rows, num_cols)
                df = finalize_dataframe(df, df_Category)
                df, category_list, difficulty_list = stock_and_remove(df)
                df_copy, num_students, num_rows, num_cols = compute_scores_and_ratios(df)
                csi = compute_csi(df, df_copy, num_students)
                cpi = compute_cpi(df, num_rows, num_cols)
                df_copy, sum_s_value = insert_and_adjust_data(df_copy, csi, cpi, num_cols,difficulty_list, category_list)
                d_row, d_col, d_ave, mc, Db, betweenlPtoS,difference_coefficient = calculate_difference_coefficient(df,sum_s_value)
                choice = st.radio("どの曲線を前面に配置しますか？", ["S曲線を前面に配置", "P曲線を前面に配置"])
                css, html_table = generate_html_table_with_css(df_copy, df, choice)
                
                # Streamlitに更新されたHTMLテーブルを表示
                # HTMLテーブルを表示
                st.write("D* = ",difference_coefficient)
                st.write(css, unsafe_allow_html=True)
                st.write(html_table, unsafe_allow_html=True)

                st.write("\n")

                if st.button("Print"):
                    html_table = f"<div><strong>D* =  {difference_coefficient}</strong></div>" + html_table    
                    html_with_css = css + html_table
                    with open("s_r_score_print.html", "w") as f:
                        f.write(html_with_css)
                    webbrowser.open('file://' + os.path.realpath("s_r_score_print.html"))

                st.write("C.S.iの分布")
                fig = create_plot(df_copy)
                st.plotly_chart(fig)
                
                st.write("C.P.iの分布")
                fig_cpi = create_cpi_plot(df_copy)
                st.plotly_chart(fig_cpi)

               
                    
            else:
                st.write(f"予期しない値 '{first_cell_value}' が1行目1列にあります.csvの1行1列は［S-Pscore］かとして下さい．")

if __name__ == "__main__":
    main()