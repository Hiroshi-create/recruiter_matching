import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_fontja
from matplotlib.backends.backend_pdf import PdfPages
import io
import os
import data_definitions as const


try:
    import config
except ImportError:
    st.error("`config.py` が見つかりません。Streamlitアプリケーションと同じディレクトリに配置されているか確認してください。")
    st.stop()



# openpyxlのインストール
# pip install openpyxl



try:
    sns.set(font="IPAexGothic", style="whitegrid")
except Exception:
    st.warning("日本語フォント（IPAexGothic）が見つかりません。グラフの文字化けが発生する可能性があります。`pip install japanize-matplotlib` をお試しください。")
    sns.set(style="whitegrid")



st.set_page_config(layout="wide")
st.title("学生・担当者 マッチング結果 可視化レポート")



st.sidebar.title("⚙️ 設定")
partner_type_selection = st.sidebar.radio(
    "分析対象を選択してください",
    ("使用データ確認", "リクルーター", "現場代表社員")
)



# --- ▼ データ読込共通部 ---
employees_file = "matching_data/employees_data.csv"
recruiters_file = "matching_data/recruiters_data.csv"
students_file = "matching_data/students_data.csv"

def clean_string_columns(df):
    if df is None:
        return None
    for col in df.select_dtypes(include=['object']).columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.replace('\u3000', ' ', regex=False).str.replace('\xa0', ' ', regex=False).str.strip()
    return df

df_students, df_recruiters, df_employees = None, None, None
try:
    df_students = pd.read_csv(students_file, na_values=[''])
    df_students = clean_string_columns(df_students)
except Exception:
    df_students = None
try:
    df_recruiters = pd.read_csv(recruiters_file, na_values=[''])
    df_recruiters = clean_string_columns(df_recruiters)
except Exception:
    df_recruiters = None
try:
    df_employees = pd.read_csv(employees_file, na_values=[''])
    df_employees = clean_string_columns(df_employees)
except Exception:
    df_employees = None



matching_results_recruiters_file = "output_results/matching_results_recruiters.csv"
matching_results_employees_file = "output_results/matching_results_employees.csv"
df_matching_recruiters, df_matching_employees = None, None
try:
    df_matching_recruiters = pd.read_csv(matching_results_recruiters_file, na_values=[''])
    df_matching_recruiters = clean_string_columns(df_matching_recruiters)
except Exception:
    df_matching_recruiters = None
try:
    df_matching_employees = pd.read_csv(matching_results_employees_file, na_values=[''])
    df_matching_employees = clean_string_columns(df_matching_employees)
except Exception:
    df_matching_employees = None



# --- ▼ マッチング結果から元データの一部を抽出するヘルパー関数 ---
def extract_source_partners_students(df, partner_type):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    partner_prefix = f"{partner_type}_"
    partner_cols = {col: col.replace(partner_prefix, '') for col in df.columns if col.startswith(partner_prefix)}
    df_src_partners = df[list(partner_cols.keys())].rename(columns=partner_cols).drop_duplicates().reset_index(drop=True)
    student_prefix = "学生_"
    student_cols = {col: col.replace(student_prefix, '') for col in df.columns if col.startswith(student_prefix)}
    df_src_students = df[list(student_cols.keys())].rename(columns=student_cols).drop_duplicates().reset_index(drop=True)
    return df_src_partners, df_src_students



# --------- マッチング結果のDataFrameを整形する関数 ---------
def prepare_match_result_for_excel(df_match, partner_type):
    if df_match is None or df_match.empty:
        return None
    df = df_match.copy()
    columns_all_nan = [col for col in df.columns if df[col].isnull().all()]
    df = df.drop(columns=columns_all_nan)

    if 'ペナルティ' in df.columns and 'マッチ度順位' not in df.columns:
        df['マッチ度順位'] = df['ペナルティ'].rank(method='min', ascending=True).astype(int)
    
    if 'マッチ度順位' in df.columns and 'ペナルティ' in df.columns:
        cols = df.columns.tolist()
        try:
            cols.remove('マッチ度順位')
            penalty_index = cols.index('ペナルティ')
            cols.insert(penalty_index + 1, 'マッチ度順位')
            df = df[cols]
        except ValueError:
            pass

    partner_kana_col = None
    partner_id_col = f"{partner_type}_社員番号"
    partner_cols_prefix = [c for c in df.columns if c.startswith(f"{partner_type}_")]
    for col in partner_cols_prefix:
        if "カナ" in col:
            partner_kana_col = col
            break

    if partner_kana_col and partner_id_col in df.columns:
        assign_count = df[partner_id_col].value_counts()
        id_to_kana = dict(zip(df[partner_id_col], df[partner_kana_col]))
        kana_to_count = {id_to_kana.get(emp_id): cnt for emp_id, cnt in assign_count.items() if id_to_kana.get(emp_id) is not None}
        new_col = df[partner_kana_col].map(kana_to_count).fillna(0).astype(int)

        if "担当学生数" in df.columns:
            df = df.drop("担当学生数", axis=1)
        insert_idx = df.columns.get_loc(partner_kana_col) + 1
        df.insert(insert_idx, "担当学生数", new_col)
        df = df.sort_values(partner_kana_col, ascending=True, kind="stable", na_position="last")

    partner_cols = [c for c in df.columns if c.startswith(f"{partner_type}_")]
    if partner_kana_col and "担当学生数" in df.columns and "担当学生数" not in partner_cols:
        p_kana_idx = partner_cols.index(partner_kana_col)
        partner_cols.insert(p_kana_idx + 1, "担当学生数")

    student_cols = [c for c in df.columns if c.startswith("学生_")]
    other_cols = [c for c in df.columns if (c not in partner_cols) and (c not in student_cols)]
    reordered_columns = partner_cols + student_cols + other_cols
    reordered_columns = [c for c in reordered_columns if c in df.columns]
    return df[reordered_columns]


# --- ▼ スタイル適用済みテーブルを作成する準備と関数 ---

def style_cells(row, partner_type, attributes_to_color, student_cols_map, partner_cols_map,
                student_unique_values, partner_unique_values, student_df_cols, partner_df_cols):
    styles = [''] * len(row)
    for attr in attributes_to_color:
        s_col, p_col = student_cols_map.get(attr), partner_cols_map.get(attr)
        if not s_col or not p_col: continue

        s_val, p_val = row.get(s_col), row.get(p_col)
        s_idx, p_idx = row.index.get_loc(s_col), row.index.get_loc(p_col)

        is_match = False
        if pd.notna(s_val) and pd.notna(p_val):
            if attr == '所属' and partner_type == 'リクルーター':
                is_match = str(s_val) in str(p_val)
            else:
                is_match = str(s_val) == str(p_val)
        
        if pd.notna(s_val):
            if is_match:
                styles[s_idx] = 'background-color: #28a745'
            else:
                is_unmatchable = (
                    attr not in partner_df_cols or pd.isna(p_val) or
                    (s_val not in partner_unique_values.get(attr, set()) and (attr != '所属' or partner_type != 'リクルーター')) or
                    (attr == '所属' and partner_type == 'リクルーター' and not any(str(s_val) in str(pv) for pv in partner_unique_values.get(attr, set())))
                )
                styles[s_idx] = 'background-color: #FF8A80' if is_unmatchable else 'background-color: #FFD700'

        if pd.notna(p_val):
            if is_match:
                styles[p_idx] = 'background-color: #28a745'
            else:
                student_check_attr = '学部名称' if attr == '所属' and partner_type == 'リクルーター' else attr
                is_unmatchable = (
                    student_check_attr not in student_df_cols or pd.isna(s_val) or
                    (p_val not in student_unique_values.get(attr, set()) and (attr != '所属' or partner_type != 'リクルーター')) or
                    (attr == '所属' and partner_type == 'リクルーター' and not any(str(sv) in str(p_val) for sv in student_unique_values.get(attr, set())))
                )
                styles[p_idx] = 'background-color: #FF8A80' if is_unmatchable else 'background-color: #FFD700'
                
    return styles

def create_styled_dataframe(df_match, partner_type, df_students, df_all_partners, weights_file):
    if df_match is None or df_match.empty:
        return None

    df_display = prepare_match_result_for_excel(df_match, partner_type)
    if df_display is None:
        return None

    try:
        df_w = pd.read_csv(weights_file, na_values=[''])
        priority_attributes = df_w.sort_values('重み', ascending=False)['属性'].tolist()
    except Exception:
        priority_attributes = []

    student_prefix = "学生_"
    partner_prefix = f"{partner_type}_"
    student_cols_map = {c.replace(student_prefix, ''): c for c in df_display.columns if c.startswith(student_prefix)}
    partner_cols_map = {c.replace(partner_prefix, ''): c for c in df_display.columns if c.startswith(partner_prefix)}
    
    attributes_to_color = [attr for attr in priority_attributes if attr in student_cols_map and attr in partner_cols_map]
    if partner_type == 'リクルーター' and '所属' in partner_cols_map and '所属' not in attributes_to_color:
        attributes_to_color.append('所属')

    student_unique_values, partner_unique_values = {}, {}
    if df_students is not None:
        for attr in attributes_to_color:
            source_attr = '学部名称' if attr == '所属' else attr
            if source_attr in df_students.columns:
                student_unique_values[attr] = set(df_students[source_attr].dropna().unique())

    if df_all_partners is not None:
        for attr in attributes_to_color:
            if attr in df_all_partners.columns:
                partner_unique_values[attr] = set(df_all_partners[attr].dropna().unique())

    student_df_cols = set(df_students.columns) if df_students is not None else set()
    partner_df_cols = set(df_all_partners.columns) if df_all_partners is not None else set()

    return df_display.style.apply(
        style_cells, axis=1,
        partner_type=partner_type, attributes_to_color=attributes_to_color,
        student_cols_map=student_cols_map, partner_cols_map=partner_cols_map,
        student_unique_values=student_unique_values, partner_unique_values=partner_unique_values,
        student_df_cols=student_df_cols, partner_df_cols=partner_df_cols
    ).format(na_rep='-')

styler_recruiters = create_styled_dataframe(
    df_matching_recruiters, "リクルーター", df_students, df_recruiters, "output_results/optimized_weights_recruiters.csv"
)
styler_employees = create_styled_dataframe(
    df_matching_employees, "現場代表社員", df_students, df_employees, "output_results/optimized_weights_employees.csv"
)

# --- ▼ サイドバー一括エクスポート用すべてのテーブル ---
sidebar_tables = {
    "学生データ": df_students,
    "リクルーターデータ": df_recruiters,
    "現場代表社員データ": df_employees,
    "リクルーター×学生マッチ結果": styler_recruiters,
    "現場代表社員×学生マッチ結果": styler_employees,
}

# --- ▼ すべてのテーブルをExcel多シートに出力する関数 (Styler対応) ---
def export_tables_to_excel(tables_dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, obj in tables_dict.items():
            if obj is None: continue
            safe_name = sheet_name[:31]
            if isinstance(obj, pd.io.formats.style.Styler):
                obj.to_excel(writer, sheet_name=safe_name, index=False)
            elif isinstance(obj, pd.DataFrame):
                obj.to_excel(writer, sheet_name=safe_name, index=False)
    out.seek(0)
    return out

# --- ▼ 「使用データ確認」ページ ---
if partner_type_selection == "使用データ確認":
    st.header("使用データ確認")
    tab1, tab2, tab3 = st.tabs(["学生データ", "リクルーターデータ", "現場代表社員データ"])
    with tab1:
        st.subheader("学生データ")
        if df_students is not None:
            st.dataframe(df_students, use_container_width=True)
        else:
            st.error("学生データの読み込みに失敗しました。")
    with tab2:
        st.subheader("リクルーターデータ")
        if df_recruiters is not None:
            st.dataframe(df_recruiters, use_container_width=True)
        else:
            st.error("リクルーターデータの読み込みに失敗しました。")
    with tab3:
        st.subheader("現場代表社員データ")
        if df_employees is not None:
            st.dataframe(df_employees, use_container_width=True)
        else:
            st.error("現場代表社員データの読み込みに失敗しました。")
    
    st.sidebar.divider()
    st.sidebar.subheader("すべてのテーブルをExcelでダウンロード")
    if all(table is not None for table in sidebar_tables.values()):
        excel_out = export_tables_to_excel(sidebar_tables)
        st.sidebar.download_button(
            label="全てのテーブルをExcelで一括ダウンロード",
            data=excel_out,
            file_name="all_tables_styled.xlsx", # ファイル名を変更
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="学生・担当者原データ3種類と各担当者種別のスタイル付きマッチング結果を全て1つのExcelファイルでダウンロードします。"
        )
    else:
        st.sidebar.info("一部のデータファイルが見つからないため、一括ダウンロードはできません。全てのCSVファイルをご用意ください。")
    st.stop()



# --- ▼ マッチング分析部 ---
if partner_type_selection == "リクルーター":
    df = df_matching_recruiters
    df_styler = styler_recruiters
    weights_file = "output_results/optimized_weights_recruiters.csv"
else:
    df = df_matching_employees
    df_styler = styler_employees
    weights_file = "output_results/optimized_weights_employees.csv"

@st.cache_data
def load_priority_attributes(weights_path):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        weights_full_path = os.path.join(base_dir, weights_path)
        df_weights = pd.read_csv(weights_full_path, na_values=[''])
        df_weights = clean_string_columns(df_weights)
        sorted_attrs = df_weights.sort_values('重み', ascending=False)['属性'].tolist()
        return df_weights, sorted_attrs, None
    except FileNotFoundError:
        return None, [], f"重みファイルが見つかりません: {weights_full_path}"
    except Exception as e:
        return None, [], f"重みファイルの読み込み中にエラーが発生しました: {e}"

df_weights, priority_attributes, error_message_weights = load_priority_attributes(weights_file)
if error_message_weights:
    st.error(f"エラー: {error_message_weights}")
    st.stop()

st.header(f"学生 対 {partner_type_selection} マッチング分析")
st.write(f"最適化計算によって得られた学生と{partner_type_selection}のマッチング結果を分析・可視化します。")

if df is None:
    st.error(f"マッチング結果ファイルが見つかりません。")
    st.info("app.pyと同じ階層に`output_results`フォルダとCSVファイルが正しく配置されているか確認してください。")
    st.stop()

st.subheader(f"属性別の重みと一致率 ({partner_type_selection})")
ordered_match_info = {}
display_order = [f"{attr}一致率" for attr in priority_attributes]
for key in display_order:
    rate = 0.0
    if key == '所属一致率' and partner_type_selection == 'リクルーター':
        s_col, p_col = '学生_学部名称', 'リクルーター_所属'
        if s_col in df.columns and p_col in df.columns:
            rate = df.apply(lambda row: pd.notna(row[s_col]) and pd.notna(row[p_col]) and row[s_col] in row[p_col], axis=1).mean()
    else:
        attribute_name = key.replace('一致率', '')
        s_col, p_col = f'学生_{attribute_name}', f'{partner_type_selection}_{attribute_name}'
        if s_col in df.columns and p_col in df.columns:
            valid_rows = df[[s_col, p_col]].dropna()
            rate = 0.0 if valid_rows.empty else (valid_rows[s_col] == valid_rows[p_col]).mean()
    ordered_match_info[key] = rate

if df_weights is not None:
    df_w = df_weights.sort_values('重み', ascending=False).set_index('属性')
    if ordered_match_info:
        match_df = pd.DataFrame.from_dict(ordered_match_info, orient='index', columns=['一致率'])
        match_df.index = match_df.index.str.replace('一致率', '')
        merged_df = df_w.join(match_df, how='left').replace(np.nan, None)
        st.write("各属性の重みと、実際のマッチング結果における一致率を可視化しています。")
        style = merged_df.style.format({"重み": "{:.3f}", "一致率": "{:.1%}"}, na_rep="-")
        if '重み' in merged_df: style = style.bar(subset=["重み"], color='#FFA07A', vmin=0)
        if '一致率' in merged_df: style = style.bar(subset=["一致率"], color='#90EE90', vmin=0, vmax=1)
        st.dataframe(style, use_container_width=True)
    else:
        st.table(df_w)
else:
    st.warning("重みデータが見つかりませんでした。")

st.subheader("マッチング結果データ")
st.markdown("""
**凡例**
- <span style="background-color:#28a745; padding: 2px 6px; border-radius: 4px;">緑色セル</span>: 学生と担当者の属性が一致している項目
- <span style="background-color:#FF8A80; padding: 2px 6px; border-radius: 4px;">赤色セル</span>: マッチング不可能な項目（相手データが存在しない、または候補にない）
- <span style="background-color:#FFD700; padding: 2px 6px; border-radius: 4px;">黄色セル</span>: マッチングしなかった項目
""", unsafe_allow_html=True)
st.write("")

if df_styler:
    df_display = df_styler.data
    columns_all_nan = [col for col in df_display.columns if df_display[col].isnull().all()]
    columns_visible = [col for col in df_display.columns if col not in columns_all_nan]
    st.dataframe(df_styler, column_order=columns_visible, use_container_width=True)
    
    excel_data = io.BytesIO()
    df_styler.to_excel(excel_data, engine='openpyxl', index=False)
    excel_data.seek(0)

    st.download_button(
        label="📊 表示中のマッチング結果をExcelでダウンロード",
        data=excel_data,
        file_name=f"matching_results_{'recruiter' if partner_type_selection == 'リクルーター' else 'employee'}_styled.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="現在表示されている色付けされたマッチング結果をExcelファイルでダウンロードします。"
    )
else:
    st.warning("表示するマッチングデータがありません。")

pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer, metadata={'Title': f'Matching Report ({partner_type_selection})'}) as pdf:
    st.subheader(f"{partner_type_selection}ごとの担当学生数（負荷状況）")
    if partner_type_selection == "リクルーター":
        df_source_partners, df_source_students = extract_source_partners_students(df, "リクルーター")
    else:
        df_source_partners, df_source_students = extract_source_partners_students(df, "現場代表社員")
    num_students, num_partners = len(df_source_students), len(df_source_partners)

    min_assignments, max_assignments = config.calculate_assignment_range(num_students, num_partners, const.ASSIGNMENT_DIFFERENCE)
    if min_assignments is not None and max_assignments is not None:
        st.info(f"**担当学生数の許容範囲:** 各{partner_type_selection}に割り当てられる学生数は、**{min_assignments}人から{max_assignments}人** の範囲で最適化されています。")
    else:
        st.warning("担当者数が0人のため、担当学生数の許容範囲は計算できません。")

    partner_id_col = f"{partner_type_selection}_社員番号"
    source_id_col = "社員番号"
    kana_col = [c for c in df.columns if c.startswith(partner_type_selection) and "カナ" in c]

    if partner_id_col in df.columns and source_id_col in df_source_partners.columns and kana_col:
        df_display_for_graph = styler_recruiters.data if partner_type_selection == "リクルーター" else styler_employees.data
        kana_col_name = kana_col[0]
        partner_id_to_kana = df_display_for_graph[[partner_id_col, kana_col_name]].drop_duplicates().set_index(partner_id_col)[kana_col_name].to_dict()
        all_partner_ids = df_source_partners[source_id_col].sort_values().unique()
        workload = df[partner_id_col].value_counts().reindex(all_partner_ids, fill_value=0)
        kana_labels = [partner_id_to_kana.get(pid, str(pid)) for pid in all_partner_ids]

        fig3, ax3 = plt.subplots(figsize=(max(10, len(all_partner_ids) * 0.4), 6))
        sns.barplot(x=kana_labels, y=workload.values, ax=ax3, palette="viridis")
        ax3.set_title(f"{partner_type_selection}ごとの担当学生数", fontsize=16)
        ax3.set_xlabel(f"{partner_type_selection} カナ氏名", fontsize=12)
        ax3.set_ylabel("担当学生数", fontsize=12)
        ax3.tick_params(axis='x', rotation=45, labelsize=10)
        for i, v in enumerate(workload.values):
            ax3.text(i, v + 0.3, str(v), ha='center', color='black')
        st.pyplot(fig3)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)
    else:
        st.warning("担当者IDカラムまたはカナ氏名カラムが見つからないため、負荷状況グラフを表示できません。")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("属性ごとの一致度グラフ")
        fig, ax = plt.subplots(figsize=(8, 6))
        if ordered_match_info:
            plot_data = pd.Series(ordered_match_info).fillna(0)
            sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax, color="#3498db")
            ax.set_title(f"属性別 マッチング一致率 ({partner_type_selection})", fontsize=14)
            ax.set_xlabel("属性", fontsize=10)
            ax.set_ylabel("一致率", fontsize=10)
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            for i, v in enumerate(plot_data.values):
                ax.text(i, v + 0.02, f"{v:.1%}", ha='center', color='black')
            st.pyplot(fig)
            pdf.savefig(fig, bbox_inches="tight")
        else:
            st.warning("一致率データがないため、グラフを表示できません。")
        plt.close(fig)
    with col2:
        st.subheader("ペナルティスコアの分布")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        if 'ペナルティ' in df.columns:
            sns.histplot(df["ペナルティ"], bins=20, kde=True, ax=ax2, color="#2ecc71")
            ax2.set_title(f"ペナルティスコアの分布 ({partner_type_selection})", fontsize=14)
            ax2.set_xlabel("ペナルティスコア", fontsize=10)
            ax2.set_ylabel("マッチング数", fontsize=10)
            st.pyplot(fig2)
            pdf.savefig(fig2, bbox_inches="tight")
        else:
            st.warning("「ペナルティ」列が見つかりません。")
        plt.close(fig2)

st.sidebar.divider()
st.sidebar.subheader("すべてのテーブルをExcelでダウンロード")
if all(table is not None for table in sidebar_tables.values()):
    excel_out_all = export_tables_to_excel(sidebar_tables)
    st.sidebar.download_button(
        label="全てのテーブルをExcelで一括ダウンロード",
        data=excel_out_all,
        file_name="all_tables_styled.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="学生・担当者原データ3種類と各担当者種別のスタイル付きマッチング結果を全て1つのExcelファイルでダウンロードします。"
    )
else:
    st.sidebar.info("一部のデータファイルが見つからないため、一括ダウンロードはできません。全てのCSVファイルをご用意ください。")
