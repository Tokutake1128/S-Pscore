import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def preprocess_dataframe(df):
    """
    DataFrameの前処理
    """
    # NaN値の処理
    if df.iloc[4:, 0].isna().all():
        df.iloc[4:, 0] = np.arange(1, len(df) - 3)
    
    # 列の結合と削除
    df.iloc[4:, 1] = df.iloc[4:, 0].astype(str) + df.iloc[4:, 1].astype(str)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.index[0])

    # 新規追加項目の処理
    df_shape_row, df_shape_col = df.shape
    df.iloc[0, 1:df_shape_col+1] = df.iloc[0, 1:df_shape_col+1].fillna("未入力")
    df.iloc[1, 1:df_shape_col+1] = df.iloc[1, 1:df_shape_col+1].fillna("未入力")
    df.iloc[2, :] = df.iloc[2, :].apply(lambda x: int(float(x)) if str(x).replace('.', '', 1).isdigit() else x)

    return df, df_shape_col

def format_and_sort_dataframe(df):
    """
    DataFrameのソートと整形を行い、合計値をセット
    """
    
    # ソートのためにdfの2行目までを最終行に追加
    first_2_rows = df.iloc[:2, :]
    df = df.iloc[2:, :]
    df = pd.concat([df, first_2_rows], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)

    num_rows, num_cols = df.shape

    cols_to_convert = [col for col in df.columns if col != '生徒/問題']
    for col in cols_to_convert:
        df.loc[:num_rows-3, col] = df.loc[:num_rows-3, col].astype(int)

    # 合計値のセット
    df['合計'] = ''
    df.loc[:num_rows-3, '合計'] = df.iloc[:num_rows-2, 1:num_cols].sum(axis=1)
    
    return df,num_rows, num_cols

def set_column_totals(df, num_rows, num_cols):
    """
    DataFrameに列合計値をセット
    """
    
    insert_index = num_rows - 2
    df_top = df.iloc[:insert_index]
    df_bottom = df.iloc[insert_index:]
    new_data = {'生徒/問題': ['列合計']} 
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df_top, new_df, df_bottom], ignore_index=True)
    df_Category = df.iloc[-2:].copy()
    df = df.iloc[:-2]
    sum_colsvalue = df.iloc[2:, 1:num_cols].sum(axis=1)
    df.iloc[2:, num_cols] = sum_colsvalue 
    temp_df = df.iloc[0:num_rows, :]
    sum_rowsvalue = temp_df.sum(axis=0)
    r_row_value = num_rows - 2
    df.iloc[r_row_value, 1:num_cols] = sum_rowsvalue[1:num_cols]
    
    return df,df_Category

def finalize_dataframe(df, df_Category):
    """
    小数点の削除と合計のソート
    """
    # 小数点の削除
    df_rsum = df.iloc[-1:].copy()
    df = df.iloc[:-1]
    df = df.sort_values(by='合計', ascending=False)
    df = pd.concat([df, df_rsum], axis=0)
    df = pd.concat([df, df_Category], axis=0)
    df.set_index('生徒/問題', inplace=True)
    s_sum_column = df['合計']
    df = df.drop(columns=['合計'])
    df.sort_values(by='列合計', axis=1, ascending=False, inplace=True)
    df['合計'] = s_sum_column
    df.reset_index(inplace=True)
    
    return df

def stock_and_remove(df):
    """
    最後の2行のデータを保存して，行をDataFrameから削除    
    """
    category_series = df.iloc[-2]
    category_list = category_series.tolist()
    
    difficulty_series = df.iloc[-1]
    difficulty_list = difficulty_series.tolist()
    
    df = df.iloc[:-2]
    
    return df, category_list, difficulty_list

def compute_scores_and_ratios(df):
    num_rows, num_cols = df.shape
    df_copy = df.copy()
    df_copy.at[num_rows, '生徒/問題'] = '%'
    df_copy.at[num_rows+1, '生徒/問題'] = 'C.P.i'
    full_marks_student = df.shape[1]-2 
    df_copy['%'] = (df_copy['合計']/(full_marks_student))*100
    r_row_percent = df_copy[df_copy['生徒/問題'] == '列合計'].index[0]
    df_copy.at[r_row_percent, '%'] = ""
    df_copy['%'] = pd.to_numeric(df_copy['%'], errors='coerce')
    df_copy['%'] = df_copy['%'].apply(lambda x: round(x, 1) if not pd.isna(x) else x)
    df_copy['Blank']= ""
    for i in range(num_rows-1):
        count_b = df_copy.iloc[i, 1:].tolist().count('b')
        df_copy.at[i, 'Blank'] = count_b
    df_copy['C.S.i']=""
    num_students = df_copy.shape[0] - 3
    num_rows2, num_cols2 = df_copy.shape
    bottom = num_rows2-3
    for i in range(num_cols2-4):
        r_value = df_copy.iat[num_rows2-3, i+1]
        r_value = int(r_value)
        if r_value != 0:
            df_copy.iat[num_rows2-2, i+1] = round((r_value/bottom) * 100,0)
        else:
            df_copy.iat[num_rows2-2, i+1] = 0
    return df_copy, num_students, num_rows, num_cols

def compute_csi(df,df_copy, num_students):
    #注意指数(C.S.i)の算出
    S_left = []
    #A=各生徒のS曲線から左の0に対応する正答者数の和
    for i in range(num_students):
        S_sum_value = df_copy.at[i, '合計']
        S_sum_value_int = int(S_sum_value + 1)
        sum_S_left = sum([df_copy.loc[num_students, df_copy.columns[j]] for j in range(S_sum_value_int) if df_copy.iat[i, j] == 0])
        S_left.append(sum_S_left)
    
    #B=その生徒のS曲線から右の1に対応する正答者数の和
    S_light = []
    for i in range(len(df)-1):
        S_sum_value = df.loc[i,'合計']
        cols_check = df.columns[S_sum_value + 1:-1]
        
        total = 0
        for col in cols_check:
            if df.loc[i,col] == 1:
                total += df.loc[df['生徒/問題']=='列合計',col].values[0]
        S_light.append(total if total != 0 else 0)
    S_light =  [int(value) for value in S_light] 
    
    #C=その生徒のS曲線から左の項目の正答者数の和
    left_correct_answer = []
    r_sum_row = df[df['生徒/問題'] == '列合計']
    for i in range(len(df)-1):
        value = df.loc[i,'合計']
        sum_value = r_sum_row.iloc[:, 1:value+1].sum(axis=1).values[0]
        left_correct_answer.append(sum_value)
    left_correct_answer = [int(sum_value)for sum_value in left_correct_answer]

    #D=その生徒の合計点はS-SUM
    total_score = df['合計'][:-1].tolist()

    #E=項目の正答者数の平均
    s_sum_total = df_copy['合計'].sum()
    num_cols_copy = df_copy.shape[1]
    s_sum_average = round(s_sum_total/(num_cols_copy-5),2)

    #C.S.i
    csi = []
    for i in range(num_students):
        denominator = round(left_correct_answer[i] - (total_score[i] * s_sum_average),2)
        if denominator == 0:
            csi.append(0)
        else:
            c_s_i_value = (S_left[i] - S_light[i]) / denominator
            csi.append(c_s_i_value)
    csi = [abs(round(value, 2)) for value in csi]

    return csi

def compute_cpi(df, num_rows, num_cols):
    # 注意指数(C.P.i)の算出

    # A = P曲線より上の0に対応する合計合計の和
    r_sum_value = df[df['生徒/問題'] == '列合計'].iloc[:, 1:-1].values[0].tolist()
    up_p_line = []
    for idx, col in enumerate(df.columns[1:-1]):
        sum_s_sum = df.iloc[:r_sum_value[idx]][df.iloc[:r_sum_value[idx]][col] == 0]['合計'].sum()
        up_p_line.append(sum_s_sum)
    up_p_line = [int(value) for value in up_p_line]

    # B = P曲線より下の１に対応する合計合計の和
    bottom_p_line = []
    for idx, col in enumerate(df.columns[1:-1]):
        sum_bottom_sum = df.iloc[r_sum_value[idx]:-1][df.iloc[r_sum_value[idx]:-1][col] == 1]['合計'].sum()
        bottom_p_line.append(sum_bottom_sum)
    bottom_p_line = [int(value) for value in bottom_p_line]

    # C = P曲線から上の生徒の合計合計の和
    total_p_score = []
    for idx, col in enumerate(df.columns[1:-1]):
        total_p_value = df.iloc[:r_sum_value[idx]]['合計'].sum()
        total_p_score.append(total_p_value)
    total_p_score = [int(value) for value in total_p_score]

    # D = 問題jの正答者数
    r_sum_value = [int(value) for value in r_sum_value]

    # E = 平均合計
    total_s_sum = df[df['生徒/問題'] != '列合計']['合計'].sum()               
    p_average = round(total_s_sum / (num_rows - 1), 2)
    
    # C.P.i
    cpi = [] 
    for i in range(num_cols - 2):
        denominator = total_p_score[i] - r_sum_value[i] * p_average
        if denominator == 0:
            cpi.append(0)
        else:
            c_p_i_value = (up_p_line[i] - bottom_p_line[i]) / denominator
            cpi.append(c_p_i_value)

    cpi = [abs(round(value, 2)) for value in cpi]

    return cpi

def insert_and_adjust_data(df_copy, csi, cpi, num_cols,difficulty_list, category_list):
    """
    df_copyに注意係数の値を挿入し、さらにデータを調整して出力を整形します。
    """
    
    # df_copyに注意係数の値を挿入
    while len(csi) < len(df_copy):
        csi.append(np.nan)
    df_copy['C.S.i'] = csi
    df_copy.set_index('生徒/問題', inplace=True)
    num_cols_from_first = len(df_copy.columns)
    cpi = cpi[:num_cols_from_first]
    while len(cpi) < num_cols_from_first:
        cpi.append(np.nan)
    df_copy.iloc[-1, :] = cpi

    # 調整
    sum_s_value = int(df_copy['合計'].sum())
    df_copy.at['列合計', '合計'] = sum_s_value
    r_percent_data = df_copy.loc['%', :df_copy.columns[num_cols-1]]
    r_percent_data = int(r_percent_data.sum())
    num_cols_bottom = num_cols-2
    r_percent_ave = round((r_percent_data / num_cols_bottom), 1)
    df_copy.at['%', '合計'] = r_percent_ave

    header_list = df_copy.columns.tolist()
    header_list.insert(0, "生徒/問題")

    df_copy.index.name = "index"

    header_cols = df_copy.shape[1]
    df_copy.columns = range(header_cols)

    index_value = header_list[0]
    df_copy.loc[index_value] = header_list[1:] + [None] * (len(df_copy.columns) - len(header_list) + 1)
    df_copy = df_copy.reindex([index_value] + [idx for idx in df_copy.index if idx != index_value])

    index_value = difficulty_list[0]
    df_copy.loc[index_value] = difficulty_list[1:] + [None] * (len(df_copy.columns) - len(difficulty_list) + 1)
    df_copy = df_copy.reindex([index_value] + [idx for idx in df_copy.index if idx != index_value])

    index_value = category_list[0]
    df_copy.loc[index_value] = category_list[1:] + [None] * (len(df_copy.columns) - len(category_list) + 1)
    df_copy = df_copy.reindex([index_value] + [idx for idx in df_copy.index if idx != index_value])

    df_copy = df_copy.fillna('')

    return df_copy, sum_s_value

def calculate_difference_coefficient(df,sum_s_value):
    d_row, d_col = df.shape
    d_row = d_row - 1
    d_col = d_col - 2
    d_ave = round(sum_s_value/(d_row * d_col),2)
    m = int((d_row * d_col)**0.5 + 0.5)
    
    listA = [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0.317, 0.321, 0.326, 0.33, 0.334, 0.337, 0.341, 0.344, 0.347, 0.35, 
            0.353, 0.355, 0.358, 0.36, 0.362, 0.364, 0.366, 0.367, 0.369, 0.37, 
            0.372, 0.373, 0.375, 0.377, 0.378, 0.38, 0.381, 0.382
            ]
    
    listB = [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0.383, 0.384, 0.385, 0.386, 0.387, 0.388, 0.389, 0.39, 0.391, 0.392, 
            0.393, 0.394, 0.395, 0.396, 0.397, 0.398, 0.399, 0.4, 0.401, 0.402, 
            0.403, 0.404, 0.404, 0.405, 0.405, 0.406, 0.407, 0.408
            ]
    
    listC = [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0.408, 0.409, 0.409, 0.41, 0.41, 0.411, 0.411, 0.412, 0.412, 0.413, 
            0.413, 0.414, 0.414, 0.415, 0.415, 0.416, 0.416, 0.417, 0.417, 0.418, 
            0.418, 0.419, 0.419, 0.419, 0.42, 0.42, 0.42
            ]

    listSum_S = df["合計"].iloc[0:d_row].tolist()
    lisrSum_P = df.iloc[-1, 1:d_col+1].tolist()

    if 18 <= m <= 45:
        mc = m + 20
        Db = listA[mc]
    elif 46 <= m <= 73:
        mc = m - 8
        Db = listB[mc]
    elif 74 <= m <= 100:
        mc =  m - 36
        Db = listC[mc]
    elif m < 18:
        mc = "out of range"
        Db = 0.317
    elif m > 100:
        mc = "out of range"
        Db = 0.42 
    betweenlPtoS = 0

    for i in range(d_row):
        betweenlPto0 = 0
        for j in range(d_col):
            if  0 <= lisrSum_P[j] - (1+i):
                betweenlPto0 = betweenlPto0 + 1

        betweenlPtoS =  betweenlPtoS + abs(betweenlPto0 - listSum_S[i])

    
    difference_coefficient = round(betweenlPtoS/(4*d_row*d_col*d_ave*(1-d_ave)*Db),3)
    
    return d_row, d_col, d_ave, mc, Db, betweenlPtoS, difference_coefficient

def generate_html_table_with_css(df_copy, df, choice):
    # HTMLテーブルの作成
    html_table = df_copy.to_html(escape=False, index=True, classes=["my-table"], header=False)

    # HTMLを解析
    soup = BeautifulSoup(html_table, 'html.parser')
    index_row = soup.find('th', text='index').find_parent('tr')
    index_row.decompose()
    html_table = str(soup)
    
    # cssの作成
    css = """
        <style>
    """
    

    if choice == "S曲線を前面に配置":

        #P曲線の生成
        table_col = df.shape[1]
        r_sum_row_index = df[df['生徒/問題'] == '列合計'].index[0]  # R-SUM行のインデックスを取得
        for i in range(table_col-2):
            r_sum_value = df.iloc[r_sum_row_index, i+1]
            css += f".my-table tr:nth-child({r_sum_value+3}) td:nth-child({i+2}) {{border-bottom: 3px dashed red;;}}\n"
        
        for i in range(table_col-3):
            r_sum_value = df.iloc[r_sum_row_index, i+1]
            r_sum_value1 = df.iloc[r_sum_row_index, i+1]
            r_sum_value2 = df.iloc[r_sum_row_index, i+2]
            difference_R = r_sum_value1 - r_sum_value2

            if difference_R != 0:
                for j in range(difference_R):
                    css += f".my-table tr:nth-child({r_sum_value+3-j}) td:nth-child({i+2}){{border-right: 3px dashed red;}}\n"

        #S曲線の生成
        table_row = df.shape[0]
        for i in range(table_row-1):
            s_sum_value = df.iloc[i, df.columns.get_loc('合計')]
            css += f".my-table tr:nth-child({i+1+3}) td:nth-child({s_sum_value+1}) {{border-right: 3px solid blue;}}\n"

            s_sum_value_first = df.at[i, '合計']
            s_sum_value_second = df.at[i + 1, '合計']
            difference = s_sum_value_first - s_sum_value_second

            if difference != 0:
                for j in range(difference):
                    if df.iloc[i+1]['生徒/問題'] != '列合計':
                        css += f".my-table tr:nth-child({i + 1+3}) td:nth-child({s_sum_value+1-j}) {{border-bottom: 3px solid blue;}}\n"
        
        
    elif choice == "P曲線を前面に配置":
            #S曲線の生成
        table_row = df.shape[0]
        for i in range(table_row-1):
            s_sum_value = df.iloc[i, df.columns.get_loc('合計')]
            css += f".my-table tr:nth-child({i+1+3}) td:nth-child({s_sum_value+1}) {{border-right: 3px solid blue;}}\n"

            s_sum_value_first = df.at[i, '合計']
            s_sum_value_second = df.at[i + 1, '合計']
            difference = s_sum_value_first - s_sum_value_second

            if difference != 0:
                for j in range(difference):
                    if df.iloc[i+1]['生徒/問題'] != '列合計':
                        css += f".my-table tr:nth-child({i + 1+3}) td:nth-child({s_sum_value+1-j}) {{border-bottom: 3px solid blue;}}\n"
        
        #P曲線の生成
        table_col = df.shape[1]
        r_sum_row_index = df[df['生徒/問題'] == '列合計'].index[0]  # R-SUM行のインデックスを取得
        for i in range(table_col-2):
            r_sum_value = df.iloc[r_sum_row_index, i+1]
            css += f".my-table tr:nth-child({r_sum_value+3}) td:nth-child({i+2}) {{border-bottom: 3px dashed red;;}}\n"
        
        for i in range(table_col-3):
            r_sum_value = df.iloc[r_sum_row_index, i+1]
            r_sum_value1 = df.iloc[r_sum_row_index, i+1]
            r_sum_value2 = df.iloc[r_sum_row_index, i+2]
            difference_R = r_sum_value1 - r_sum_value2

            if difference_R != 0:
                for j in range(difference_R):
                    css += f".my-table tr:nth-child({r_sum_value+3-j}) td:nth-child({i+2}){{border-right: 3px dashed red;}}\n"
    css += """
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }
        .my-table td, .my-table th {
              white-space: nowrap;
        }
        """
    css += "</style>"
    
    return css, html_table

