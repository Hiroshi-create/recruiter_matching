import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_fontja  # 日本語表示のためにインポート
from matplotlib.backends.backend_pdf import PdfPages
import io
import os

# openpyxlのインストール（styler.to_excelに必要）
# pip install openpyxl

# japanize_matplotlibが提供するフォントを設定し、グラフの日本語文字化けを解消
try:
    sns.set(font="IPAexGothic", style="whitegrid")
except Exception:
    st.warning("日本語フォント（IPAexGothic）が見つかりません。グラフの文字化けが発生する可能性があります。`pip install japanize-matplotlib` をお試しください。")
    sns.set(style="whitegrid")

st.set_page_config(layout="wide")
st.title("学生・担当者 マッチング結果 可視化レポート")

# --- サイドバーで分析対象を選択 ---
st.sidebar.title("⚙️ 設定")
partner_type_selection = st.sidebar.radio(
    "分析対象を選択してください",
    ("リクルーター", "現場社員")
)

# --- 重みファイルから優先度リストとDataFrameを動的に読み込む関数 ---
@st.cache_data
def load_priority_attributes(weights_path):
    """重み付けファイルを読み込み、DataFrameと重みの降順の属性リストを返す"""
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

# --- 選択に応じた設定と、重みファイルに基づく優先度リストの定義 ---
if partner_type_selection == "リクルーター":
    results_file = "output_results/matching_results_recruiters.csv"
    weights_file = "output_results/optimized_weights_recruiters.csv"
else:  # 現場社員
    results_file = "output_results/matching_results_employees.csv"
    weights_file = "output_results/optimized_weights_employees.csv"

df_weights, priority_attributes, error_message_weights = load_priority_attributes(weights_file)

if error_message_weights:
    st.error(f"エラー: {error_message_weights}")
    st.stop()

st.header(f"学生 対 {partner_type_selection} マッチング分析")
st.write(f"最適化計算によって得られた学生と{partner_type_selection}のマッチング結果を分析・可視化します。")


# --- データ読み込み（修正箇所） ---
@st.cache_data
def load_data(results_path, partner_type):
    """結果ファイルを読み込み、マッチング結果、パートナーソース、学生ソースのDataFrameを返す"""
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    results_full_path = os.path.join(base_dir, results_path)
    
    try:
        df_res = pd.read_csv(results_full_path)
        
        # パートナーの全候補者リストを作成
        partner_prefix = f"{partner_type}_"
        partner_cols = {col: col.replace(partner_prefix, '') for col in df_res.columns if col.startswith(partner_prefix)}
        df_src_partners = df_res[list(partner_cols.keys())].rename(columns=partner_cols).drop_duplicates().reset_index(drop=True)

        # 学生の全候補者リストを作成
        student_prefix = "学生_"
        student_cols = {col: col.replace(student_prefix, '') for col in df_res.columns if col.startswith(student_prefix)}
        df_src_students = df_res[list(student_cols.keys())].rename(columns=student_cols).drop_duplicates().reset_index(drop=True)

        return df_res, df_src_partners, df_src_students, None
    except FileNotFoundError:
        return None, None, None, f"結果ファイルが見つかりません: {results_full_path}。 スクリプトと同じ階層に`output_results`フォルダを配置してください。"
    except Exception as e:
        return None, None, None, f"ファイルの読み込み中にエラーが発生しました: {e}"

# --- データをロード（修正箇所） ---
df, df_source_partners, df_source_students, error_message = load_data(results_file, partner_type_selection)

if error_message:
    st.error(f"エラー: {error_message}")
    st.info("app.pyと同じ階層に`output_results`フォルダとCSVファイルが正しく配置されているか確認してください。")
    st.stop()

# --- ペナルティランクを追加 ---
if 'ペナルティ' in df.columns:
    df['ペナルティランク'] = df['ペナルティ'].rank(method='min', ascending=True).astype(int)
    cols = df.columns.tolist()
    if 'ペナルティランク' in cols:
        try:
            penalty_index = cols.index('ペナルティ')
            cols.insert(penalty_index + 1, cols.pop(cols.index('ペナルティランク')))
            df = df[cols]
        except ValueError:
            pass

# --- マッチング項目をハイライトする関数（修正箇所） ---
def highlight_matches(row, df_partners_source, df_students_source, partner_type):
    """
    行を受け取り、スタイルを返す関数。
    - 一致する属性を緑色にハイライト
    - 相手方の候補リストに存在しない属性値を赤色にハイライト
    """
    styles = [''] * len(row)
    highlight_style = 'background-color: #28a745; color: white;'
    impossible_style = 'background-color: #dc3545; color: white;'

    attributes_to_check = ['大学名称', '学部名称', '学科名称', '文理区分', '応募者区分', '性別', '性格タイプ', '地域']

    for attr in attributes_to_check:
        s_col = f'学生_{attr}'
        p_col = f'{partner_type}_{attr}'
        src_col = attr  # '大学名称'などの基本属性名

        if s_col not in row.index or p_col not in row.index:
            continue

        s_col_idx = row.index.get_loc(s_col)
        p_col_idx = row.index.get_loc(p_col)
        student_value = row[s_col]
        partner_value = row[p_col]

        is_student_impossible = False
        is_partner_impossible = False

        # Check 1: 学生の属性値が、全担当者候補に存在しないか
        if src_col in df_partners_source.columns and pd.notna(student_value):
            if student_value not in df_partners_source[src_col].dropna().unique():
                is_student_impossible = True
        
        # Check 2: 担当者の属性値が、全学生候補に存在しないか
        if src_col in df_students_source.columns and pd.notna(partner_value):
            if partner_value not in df_students_source[src_col].dropna().unique():
                is_partner_impossible = True

        # スタイル適用
        if pd.notna(student_value) and student_value == partner_value:
            if not is_student_impossible and not is_partner_impossible:
                styles[s_col_idx] = highlight_style
                styles[p_col_idx] = highlight_style

        if is_student_impossible:
            styles[s_col_idx] = impossible_style
        
        if is_partner_impossible:
            styles[p_col_idx] = impossible_style

    # リクルーターの「所属」に関する特別処理
    if partner_type == 'リクルーター':
        s_faculty_col, p_dept_col = '学生_学部名称', 'リクルーター_所属'
        if s_faculty_col in row.index and p_dept_col in row.index:
            if pd.notna(row[s_faculty_col]) and pd.notna(row[p_dept_col]) and str(row[s_faculty_col]) in str(row[p_dept_col]):
                s_fac_idx = row.index.get_loc(s_faculty_col)
                p_dep_idx = row.index.get_loc(p_dept_col)
                if styles[s_fac_idx] != impossible_style and styles[p_dep_idx] != impossible_style:
                    styles[s_fac_idx] = highlight_style
                    styles[p_dep_idx] = highlight_style
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

# --- 凡例とテーブル表示（修正箇所） ---
st.markdown("""
**凡例**
- <span style="color:white; background-color:#28a745; padding: 2px 6px; border-radius: 4px;">緑色セル</span>: 学生と担当者の属性が一致している項目
- <span style="color:white; background-color:#dc3545; padding: 2px 6px; border-radius: 4px;">赤色セル</span>: 学生または担当者の属性が、相手方の全候補者リストに存在しないためマッチング不可能な項目
""", unsafe_allow_html=True)
st.write("")

styled_df = df.style.apply(
    highlight_matches,
    axis=1,
    df_partners_source=df_source_partners,
    df_students_source=df_source_students,
    partner_type=partner_type_selection
)

st.dataframe(styled_df)

@st.cache_data
def to_excel_with_style(_df_styled):
    output = io.BytesIO()
    _df_styled.to_excel(output, engine='openpyxl', index=False)
    return output.getvalue()

excel_data = to_excel_with_style(styled_df)

st.download_button(
    label="📊 マッチング結果をExcelでダウンロード",
    data=excel_data,
    file_name=f"matching_results_{'recruiter' if partner_type_selection == 'リクルーター' else 'employee'}_styled.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="ハイライトされたセルを含むマッチング結果データをExcelファイルでダウンロードします。"
)

# --- PDF作成とグラフ描画 ---
pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer, metadata={'Title': f'Matching Report ({partner_type_selection})'}) as pdf:
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

    st.divider()

    st.subheader(f"{partner_type_selection}ごとの担当学生数（負荷状況）")
    partner_id_col = f"{partner_type_selection}_社員番号"
    source_id_col = "社員番号"
    if partner_id_col in df.columns and source_id_col in df_source_partners.columns:
        all_partner_ids = df_source_partners[source_id_col].sort_values().unique()
        workload = df[partner_id_col].value_counts().reindex(all_partner_ids, fill_value=0)

        fig3, ax3 = plt.subplots(figsize=(max(10, len(all_partner_ids) * 0.4), 6))
        sns.barplot(x=workload.index, y=workload.values, ax=ax3, palette="viridis")
        ax3.set_title(f"{partner_type_selection}ごとの担当学生数", fontsize=16)
        ax3.set_xlabel(f"{partner_type_selection} 社員番号", fontsize=12)
        ax3.set_ylabel("担当学生数", fontsize=12)
        ax3.tick_params(axis='x', rotation=45, labelsize=10)
        for i, v in enumerate(workload.values):
            ax3.text(i, v + 0.3, str(v), ha='center', color='black')
        st.pyplot(fig3)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)
    else:
        st.warning("担当者IDカラムが見つからないため、負荷状況グラフを表示できません。")

st.sidebar.divider()
st.sidebar.subheader("レポートのダウンロード")
pdf_buffer.seek(0)
st.sidebar.download_button(
    label=f"{partner_type_selection}のレポートをPDFで保存",
    data=pdf_buffer,
    file_name=f"matching_report_{'recruiter' if partner_type_selection == 'リクルーター' else 'employee'}.pdf",
    mime="application/pdf",
    help="現在表示されている全てのグラフを含むPDFファイルをダウンロードします。"
)
