import streamlit as st
import pandas as pd
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

# --- ▼ マッチングセル色付け関数（df, partner_type, source_dfs を受け取って style返す） ---
def highlight_matches(row, df_partners_source, df_students_source, partner_type):
    ncols = len(row)
    styles = [''] * ncols
    highlight_style = 'background-color: #28a745;'   # 緑
    impossible_style = 'background-color: #FF8A80;' # 赤
    no_match_style = 'background-color: #FFD700;'   # 黄色
    attributes_to_check = ['大学名称', '学部名称', '学科名称', '文理区分', '応募者区分', '性別', '性格タイプ', '地域']
    for attr in attributes_to_check:
        s_col = f'学生_{attr}'
        p_col = f'{partner_type}_{attr}'
        src_col = attr
        if s_col not in row.index or p_col not in row.index:
            continue
        s_col_idx = row.index.get_loc(s_col)
        p_col_idx = row.index.get_loc(p_col)
        student_value = row[s_col]
        partner_value = row[p_col]
        is_student_impossible = False
        is_partner_impossible = False
        if src_col in df_partners_source.columns and pd.notna(student_value):
            if student_value not in df_partners_source[src_col].dropna().unique():
                is_student_impossible = True
        if src_col in df_students_source.columns and pd.notna(partner_value):
            if partner_value not in df_students_source[src_col].dropna().unique():
                is_partner_impossible = True
        if pd.notna(student_value) and student_value == partner_value:
            if not is_student_impossible and not is_partner_impossible:
                styles[s_col_idx] = highlight_style
                styles[p_col_idx] = highlight_style
        elif is_student_impossible:
            styles[s_col_idx] = impossible_style
        elif is_partner_impossible:
            styles[p_col_idx] = impossible_style
        else:
            styles[s_col_idx] = no_match_style
            styles[p_col_idx] = no_match_style
    return styles

# --- ▼ データ読込共通部 ---
employees_file = "matching_data/employees_data.csv"
recruiters_file = "matching_data/recruiters_data.csv"
students_file   = "matching_data/students_data.csv"

df_students, df_recruiters, df_employees = None, None, None
try:
    df_students = pd.read_csv(students_file)
except Exception:
    df_students = None
try:
    df_recruiters = pd.read_csv(recruiters_file)
except Exception:
    df_recruiters = None
try:
    df_employees = pd.read_csv(employees_file)
except Exception:
    df_employees = None

matching_results_recruiters_file  = "output_results/matching_results_recruiters.csv"
matching_results_employees_file   = "output_results/matching_results_employees.csv"
df_matching_recruiters, df_matching_employees = None, None
try:
    df_matching_recruiters = pd.read_csv(matching_results_recruiters_file)
except Exception:
    df_matching_recruiters = None
try:
    df_matching_employees = pd.read_csv(matching_results_employees_file)
except Exception:
    df_matching_employees = None

# --- ▼ マッチングセル色付け用パートナー,学生DF抽出 helpers ---
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

df_recruiter_partners, df_recruiter_students = extract_source_partners_students(df_matching_recruiters, "リクルーター")
df_employee_partners, df_employee_students   = extract_source_partners_students(df_matching_employees, "現場代表社員")

# --- ▼ サイドバー一括エクスポート用すべてのテーブル(あとでstyle) ---
sidebar_tables = {
    "学生データ": df_students,
    "リクルーターデータ": df_recruiters,
    "現場代表社員データ": df_employees,
    "リクルーター×学生マッチ結果": None,   # スタイル付与後に詰める
    "現場代表社員×学生マッチ結果": None,
}

# --------- MATCH用色付きDataFrameを生成（pandas style）---------
def prepare_match_result_for_excel(df_match, partner_type, partners_source, students_source):
    """
    - 全カラムが空（全NaN）の列を除外
    - 「マッチ度順位」列がなければ、ペナルティ昇順で追加
    - 推奨列順で並び替え
    - highlight_matchesでセルstyle適用
    """
    if df_match is None or df_match.empty:
        return None
    df = df_match.copy()

    # 全NaN列を除外
    columns_all_nan = [col for col in df.columns if df[col].isnull().all()]
    df = df.drop(columns=columns_all_nan)

    # 「マッチ度順位」なければ作成
    if 'ペナルティ' in df.columns and 'マッチ度順位' not in df.columns:
        df['マッチ度順位'] = df['ペナルティ'].rank(method='min', ascending=True).astype(int)
        # 「マッチ度順位」を「ペナルティ」のすぐ右に
        cols = df.columns.tolist()
        penalty_index = cols.index('ペナルティ')
        cols.insert(penalty_index + 1, cols.pop(cols.index('マッチ度順位')))
        df = df[cols]
    elif 'マッチ度順位' in df.columns:
        # 既に存在すれば順序だけ整える
        cols = df.columns.tolist()
        penalty_index = cols.index('ペナルティ') if 'ペナルティ' in cols else -1
        if penalty_index != -1 and 'マッチ度順位' in cols:
            cols.insert(penalty_index + 1, cols.pop(cols.index('マッチ度順位')))
            df = df[cols]
    # 列順
    partner_cols = [c for c in df.columns if c.startswith(f"{partner_type}_")]
    student_cols = [c for c in df.columns if c.startswith("学生_")]
    other_cols = [c for c in df.columns if not (c.startswith(f"{partner_type}_") or c.startswith("学生_"))]
    reordered_columns = partner_cols + student_cols + other_cols
    # 存在するものだけで並び替え
    reordered_columns = [c for c in reordered_columns if c in df.columns]
    df_disp = df[reordered_columns]
    # style
    styled = df_disp.style.apply(
        highlight_matches,
        axis=1,
        df_partners_source=partners_source,
        df_students_source=students_source,
        partner_type=partner_type
    )
    return styled

sidebar_tables["リクルーター×学生マッチ結果"] = (
    prepare_match_result_for_excel(df_matching_recruiters, "リクルーター", df_recruiter_partners, df_recruiter_students)
    if df_matching_recruiters is not None and not df_matching_recruiters.empty else None
)
sidebar_tables["現場代表社員×学生マッチ結果"] = (
    prepare_match_result_for_excel(df_matching_employees, "現場代表社員", df_employee_partners, df_employee_students)
    if df_matching_employees is not None and not df_matching_employees.empty else None
)

# --- ▼ すべてのテーブルをExcel多シート（必要なものはstyleで色付き！） ---
def export_tables_to_excel(tables_dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, df_obj in tables_dict.items():
            if df_obj is not None:
                safe_name = sheet_name[:31]
                # DataFrameかStylerか判定して出し分け
                if isinstance(df_obj, pd.io.formats.style.Styler):
                    df_obj.to_excel(writer, sheet_name=safe_name, index=False)
                elif isinstance(df_obj, pd.DataFrame) and not df_obj.empty:
                    df_obj.to_excel(writer, sheet_name=safe_name, index=False)
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
    # --- ▼ サイドバー下部で必ずダウンロードボタンを表示 ---
    st.sidebar.divider()
    st.sidebar.subheader("すべてのテーブルをExcelでダウンロード")
    usable_tables = [k for k, df in sidebar_tables.items() if df is not None and (not isinstance(df, pd.DataFrame) or not df.empty)]
    if len(usable_tables) == len(sidebar_tables):
        excel_out = export_tables_to_excel(sidebar_tables)
        st.sidebar.download_button(
            label="全てのテーブルをExcelで一括ダウンロード",
            data=excel_out,
            file_name="all_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="学生・担当者原データ3種類と各担当者種別のマッチング結果を全て1つのExcelファイルでダウンロードします（シート分割、マッチ結果はセル色付き）。"
        )
    else:
        st.sidebar.info("全データが揃っていません。全てのCSVファイルをご用意ください。")
    st.stop()

# --- ▼ マッチング分析部 ---
@st.cache_data
def load_priority_attributes(weights_path):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        weights_full_path = os.path.join(base_dir, weights_path)
        df_weights = pd.read_csv(weights_full_path)
        sorted_attrs = df_weights.sort_values('重み', ascending=False)['属性'].tolist()
        return df_weights, sorted_attrs, None
    except FileNotFoundError:
        return None, [], f"重みファイルが見つかりません: {weights_full_path}。 スクリプトと同じ階層に`output_results`フォルダを配置してください。"
    except Exception as e:
        return None, [], f"重みファイルの読み込み中にエラーが発生しました: {e}"

if partner_type_selection == "リクルーター":
    results_file = matching_results_recruiters_file
    weights_file = "output_results/optimized_weights_recruiters.csv"
else:
    results_file = matching_results_employees_file
    weights_file = "output_results/optimized_weights_employees.csv"

df_weights, priority_attributes, error_message_weights = load_priority_attributes(weights_file)
if error_message_weights:
    st.error(f"エラー: {error_message_weights}")
    st.stop()

st.header(f"学生 対 {partner_type_selection} マッチング分析")
st.write(f"最適化計算によって得られた学生と{partner_type_selection}のマッチング結果を分析・可視化します。")

@st.cache_data
def load_data(results_path, partner_type):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    results_full_path = os.path.join(base_dir, results_path)
    try:
        df_res = pd.read_csv(results_full_path)
        partner_prefix = f"{partner_type}_"
        partner_cols = {col: col.replace(partner_prefix, '') for col in df_res.columns if col.startswith(partner_prefix)}
        df_src_partners = df_res[list(partner_cols.keys())].rename(columns=partner_cols).drop_duplicates().reset_index(drop=True)
        student_prefix = "学生_"
        student_cols = {col: col.replace(student_prefix, '') for col in df_res.columns if col.startswith(student_prefix)}
        df_src_students = df_res[list(student_cols.keys())].rename(columns=student_cols).drop_duplicates().reset_index(drop=True)
        return df_res, df_src_partners, df_src_students, None
    except FileNotFoundError:
        return None, None, None, f"結果ファイルが見つかりません: {results_full_path}。 スクリプトと同じ階層に`output_results`フォルダを配置してください。"
    except Exception as e:
        return None, None, None, f"ファイルの読み込み中にエラーが発生しました: {e}"

df, df_source_partners, df_source_students, error_message = load_data(results_file, partner_type_selection)
if error_message:
    st.error(f"エラー: {error_message}")
    st.info("app.pyと同じ階層に`output_results`フォルダとCSVファイルが正しく配置されているか確認してください。")
    st.stop()

if 'ペナルティ' in df.columns:
    df['マッチ度順位'] = df['ペナルティ'].rank(method='min', ascending=True).astype(int)
    cols = df.columns.tolist()
    if 'マッチ度順位' in cols:
        try:
            penalty_index = cols.index('ペナルティ')
            cols.insert(penalty_index + 1, cols.pop(cols.index('マッチ度順位')))
            df = df[cols]
        except ValueError:
            pass

def highlight_matches(row, df_partners_source, df_students_source, partner_type):
    ncols = len(row)
    styles = [''] * ncols
    highlight_style = 'background-color: #28a745;'   # 緑
    impossible_style = 'background-color: #FF8A80;' # 赤
    no_match_style = 'background-color: #FFD700;'   # 黄色
    attributes_to_check = ['大学名称', '学部名称', '学科名称', '文理区分', '応募者区分', '性別', '性格タイプ', '地域']
    for attr in attributes_to_check:
        s_col = f'学生_{attr}'
        p_col = f'{partner_type}_{attr}'
        src_col = attr
        if s_col not in row.index or p_col not in row.index:
            continue
        s_col_idx = row.index.get_loc(s_col)
        p_col_idx = row.index.get_loc(p_col)
        student_value = row[s_col]
        partner_value = row[p_col]
        is_student_impossible = False
        is_partner_impossible = False
        if src_col in df_partners_source.columns and pd.notna(student_value):
            if student_value not in df_partners_source[src_col].dropna().unique():
                is_student_impossible = True
        if src_col in df_students_source.columns and pd.notna(partner_value):
            if partner_value not in df_students_source[src_col].dropna().unique():
                is_partner_impossible = True
        if pd.notna(student_value) and student_value == partner_value:
            if not is_student_impossible and not is_partner_impossible:
                styles[s_col_idx] = highlight_style
                styles[p_col_idx] = highlight_style
        elif is_student_impossible:
            styles[s_col_idx] = impossible_style
        elif is_partner_impossible:
            styles[p_col_idx] = impossible_style
        else:
            styles[s_col_idx] = no_match_style
            styles[p_col_idx] = no_match_style
    return styles

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
        s_col = f'学生_{attribute_name}'
        p_col = f'{partner_type_selection}_{attribute_name}'
        if s_col in df.columns and p_col in df.columns:
            valid_rows = df[[s_col, p_col]].dropna()
            rate = 0.0 if valid_rows.empty else (valid_rows[s_col] == valid_rows[p_col]).mean()
    ordered_match_info[key] = rate

if df_weights is not None:
    df_w = df_weights.sort_values('重み', ascending=False).set_index('属性')
    if ordered_match_info:
        match_df = pd.DataFrame.from_dict(ordered_match_info, orient='index', columns=['一致率'])
        match_df.index = match_df.index.str.replace('一致率', '')
        merged_df = df_w.join(match_df, how='left')
        st.write("各属性の重みと、実際のマッチング結果における一致率を可視化しています。")
        st.dataframe(
            merged_df.style.format({"重み": "{:.3f}", "一致率": "{:.1%}"})
            .bar(subset=["重み"], color='#FFA07A')
            .bar(subset=["一致率"], color='#90EE90'),
            use_container_width=True
        )
    else:
        st.warning("一致率データが見つかりませんでした。")
        st.table(df_w)
else:
    st.warning("重みデータが見つかりませんでした。")

st.subheader("マッチング結果データ")
st.markdown("""
**凡例**
- <span style="background-color:#28a745; padding: 2px 6px; border-radius: 4px;">緑色セル</span>: 学生と担当者の属性が一致している項目
- <span style="background-color:#FF8A80; padding: 2px 6px; border-radius: 4px;">赤色セル</span>: 学生または担当者の属性が、相手方の全候補者リストに存在しないためマッチング不可能な項目
- <span style="background-color:#FFD700; padding: 2px 6px; border-radius: 4px;">黄色セル</span>: マッチングしなかった項目（比較対象が赤色セル以外）
""", unsafe_allow_html=True)
st.write("")

partner_cols = [c for c in df.columns if c.startswith(f"{partner_type_selection}_")]
student_cols = [c for c in df.columns if c.startswith("学生_")]
other_cols = [c for c in df.columns if not (c.startswith(f"{partner_type_selection}_") or c.startswith("学生_"))]
reordered_columns = partner_cols + student_cols + other_cols
df_display = df[reordered_columns]

kana_col = None
for c in partner_cols:
    if "カナ" in c:
        kana_col = c
        break

if kana_col is not None and kana_col in df_display.columns:
    partner_id_col = f"{partner_type_selection}_社員番号"
    if partner_id_col in df_display.columns:
        assign_count = df[partner_id_col].value_counts()
        id_to_kana = dict(zip(df_display[partner_id_col], df_display[kana_col]))
        kana_to_count = {}
        for emp_id, cnt in assign_count.items():
            kana_name = id_to_kana.get(emp_id)
            if kana_name is not None:
                kana_to_count[kana_name] = cnt
        df_display.insert(df_display.columns.get_loc(kana_col)+1, "担当学生数",
                          df_display[kana_col].map(kana_to_count).fillna(0).astype(int))
    df_display = df_display.sort_values(kana_col, ascending=True, kind="stable", na_position="last")

columns_all_nan = [col for col in df_display.columns if df_display[col].isnull().all()]
columns_visible = [col for col in df_display.columns if col not in columns_all_nan]

styled_df = df_display.style.apply(
    highlight_matches,
    axis=1,
    df_partners_source=df_source_partners,
    df_students_source=df_source_students,
    partner_type=partner_type_selection
)
st.dataframe(
    styled_df,
    column_order=columns_visible,
    use_container_width=True
)

@st.cache_data
def to_excel_with_style(_df_styled):
    output = io.BytesIO()
    _df_styled.to_excel(output, engine='openpyxl', index=False)
    return output.getvalue()

excel_data = to_excel_with_style(df_display)
st.download_button(
    label="📊 マッチング結果をExcelでダウンロード",
    data=excel_data,
    file_name=f"matching_results_{'recruiter' if partner_type_selection == 'リクルーター' else 'employee'}_styled.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="ハイライトされたセルを含むマッチング結果データをExcelファイルでダウンロードします。"
)

pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer, metadata={'Title': f'Matching Report ({partner_type_selection})'}) as pdf:
    st.subheader(f"{partner_type_selection}ごとの担当学生数（負荷状況）")
    num_students = len(df)
    num_partners = len(df_source_partners)
    min_assignments, max_assignments = config.calculate_assignment_range(
        num_students, num_partners, const.ASSIGNMENT_DIFFERENCE
    )
    if min_assignments is not None and max_assignments is not None:
        st.info(
            f"**担当学生数の許容範囲:** "
            f"各{partner_type_selection}に割り当てられる学生数は、"
            f"**{min_assignments}人から{max_assignments}人** の範囲で最適化されています。"
        )
    else:
        st.warning("担当者数が0人のため、担当学生数の許容範囲は計算できません。")
    partner_id_col = f"{partner_type_selection}_社員番号"
    source_id_col = "社員番号"
    if partner_id_col in df.columns and source_id_col in df_source_partners.columns and kana_col is not None and kana_col in df_display.columns:
        partner_id_to_kana = df_display[[partner_id_col, kana_col]].drop_duplicates().set_index(partner_id_col)[kana_col].to_dict()
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
            plot_data = ordered_match_info
            sns.barplot(x=list(plot_data.keys()), y=list(plot_data.values()), ax=ax, color="#3498db")
            ax.set_title(f"属性別 マッチング一致率 ({partner_type_selection})", fontsize=14)
            ax.set_xlabel("属性", fontsize=10)
            ax.set_ylabel("一致率", fontsize=10)
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            for i, v in enumerate(plot_data.values()):
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

# --- ▼ サイドバー下部で必ずダウンロードボタンを表示 ---
st.sidebar.divider()
st.sidebar.subheader("すべてのテーブルをExcelでダウンロード")
usable_tables = [k for k, df in sidebar_tables.items() if df is not None and (not isinstance(df, pd.DataFrame) or not df.empty)]
if len(usable_tables) == len(sidebar_tables):
    excel_out = export_tables_to_excel(sidebar_tables)
    st.sidebar.download_button(
        label="全てのテーブルをExcelで一括ダウンロード",
        data=excel_out,
        file_name="all_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="学生・担当者原データ3種類と各担当者種別のマッチング結果を全て1つのExcelファイルでダウンロードします（シート分割、マッチ結果はセル色付き）。"
    )
else:
    st.sidebar.info("全データが揃っていません。全てのCSVファイルをご用意ください。")