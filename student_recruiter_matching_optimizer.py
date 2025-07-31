# $ python student_recruiter_matching_optimizer.py

# -*- coding: utf-8 -*-

"""

学生・現場代表社員・リクルーターマッチングシステム

ステップ2: 最適化問題の定式化と解法 (改良版)

このプログラムは、matching_dataフォルダ内のCSVファイル

（students_data.csv, employees_data.csv, recruiters_data.csv）を読み込み、

01整数計画問題を用いて学生と担当社員（リクルーターまたは現場代表社員）の最適マッチングを2種類行います。

属性の一致度を最大化しながら、担当者の負荷を動的に計算された範囲で分散させることを目標とします。

最終的なアウトプットは output_results フォルダ内にCSVとして個別に保存します。

必要なライブラリのインストール:

pip install pandas numpy pulp matplotlib seaborn scipy

"""

import pandas as pd
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import data_definitions as const
import config

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
    sns.set_style("whitegrid", {'font.family': ['IPAexGothic']})
except Exception:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set_style("whitegrid")


class StudentRecruitingOptimizerFromCSV:
    """CSVからデータを読み込み最適マッチングを行うクラス"""
    
    def __init__(self,
                 student_csv='matching_data/students_data.csv',
                 recruiter_csv='matching_data/recruiters_data.csv',
                 employee_csv='matching_data/employees_data.csv',
                 output_folder='output_results',
                 seed=42):
        """
        初期化
        """
        np.random.seed(seed)
        self.penalty_weights = {}

        # --- データの読み込み (日本語列名のまま) ---
        self.df_students_orig = pd.read_csv(student_csv)
        self.df_recruiters_orig = pd.read_csv(recruiter_csv)
        self.df_employees_orig = pd.read_csv(employee_csv)

        # カラム名「東or西」「地域」を「地域」に統一
        for df in [self.df_students_orig, self.df_recruiters_orig, self.df_employees_orig]:
            rename_dict = {}
            if '東or西' in df.columns:
                rename_dict['東or西'] = '地域'
            if '勤務地' in df.columns:
                rename_dict['勤務地'] = '地域'
            if rename_dict:
                df.rename(columns=rename_dict, inplace=True)
                
        # 学生データにマッチング用の「地域」列を追加 (「地域」から変更)
        self.df_students = self.df_students_orig.copy()
        
        # 社員データに「地域」カラムが存在するか確認
        if '地域' in self.df_employees_orig.columns:
            locations = self.df_employees_orig['地域'].unique()
            if len(locations) > 0:
                self.df_students['地域'] = np.random.choice(locations, size=len(self.df_students))
            else:
                self.df_students['地域'] = '不明'
        else:
            # 社員データに地域情報がない場合
            self.df_students['地域'] = '不明'

        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # 現在のマッチング対象を保持する変数
        self.partner_df = None
        self.partner_name_jp = None
        self.ordered_keys = []

    def set_partner(self, partner_type):
        """
        マッチング対象のパートナー（リクルーターか現場代表社員か）を設定する。
        """
        # 1. マッチングに【使用しない】項目を定義
        exclude_keys = ['応募者コード', 'カナ氏名', '合否']

        # 2. 学生CSVの列順序から、除外項目を除いたリストを動的に生成
        base_ordered_keys = [col for col in self.df_students.columns if col not in exclude_keys]
        
        # 3. '地域'をペナルティ計算用の抽象キー'地域'に置き換える
        self.ordered_keys = ['地域' if key == '地域' else key for key in base_ordered_keys]
        
        if partner_type == 'recruiter':
            self.partner_df = self.df_recruiters_orig.copy()
            self.partner_name_jp = 'リクルーター'
            # 4. リクルーターデータに存在しない属性を優先順位から自動的に除外
            self.ordered_keys = [key for key in self.ordered_keys if key in self.partner_df.columns or key == '地域']
            
        elif partner_type == 'employee':
            self.partner_df = self.df_employees_orig.copy()
            self.partner_name_jp = '現場代表社員'
            # 4. 現場代表社員データに存在しない属性を優先順位から自動的に除外
            self.ordered_keys = [key for key in self.ordered_keys if key in self.partner_df.columns or key == '地域']
        else:
            raise ValueError("無効なパートナータイプです。'recruiter' または 'employee' を指定してください。")

        print(f"\n--- マッチング対象: {self.partner_name_jp} ---")
        priority_str = " > ".join(self.ordered_keys)
        print(f"マッチング優先度: {priority_str}")
        
        if '社員番号' not in self.partner_df.columns:
            raise ValueError(f"{self.partner_name_jp}データにIDカラム（社員番号）が見つかりません。")

    def optimize_penalty_weights(self, iterations=10, initial_base_weight=200, learning_rate=1.2, assignment_diff=10):
        """
        ペナルティ重みを反復的な探索により自動で最適化する。
        """
        if self.partner_df is None:
            print("エラー: 先に set_partner() でマッチング対象を設定してください。")
            return

        print(f"\n=== [{self.partner_name_jp}] ペナルティ重みの自動最適化開始 ===")
        
        weights = {}
        current_weight = float(initial_base_weight)
        for key in self.ordered_keys:
            weights[key] = current_weight
            current_weight = max(1.0, current_weight / 2.0)
        self.penalty_weights = weights

        best_weights = self.penalty_weights.copy()
        best_total_match_rate = -1.0

        for i in range(iterations):
            print(f"\n--- イテレーション {i+1}/{iterations} ---")
            print(f"現在の試行重み: { {k: round(v, 2) for k, v in self.penalty_weights.items()} }")
            
            results, _ = self.solve_optimization(assignment_diff=assignment_diff, verbose=False)
            
            if results is None:
                print("この重みセットでは最適解が見つかりませんでした。重みを調整して続行します。")
                self.penalty_weights = {k: v * 0.9 for k, v in self.penalty_weights.items()}
                continue
                
            analysis = self.analyze_results()
            if analysis is None:
                continue

            match_rates = {key: analysis[key] for key in self.ordered_keys if key in analysis}
            current_total_match_rate = np.mean(list(match_rates.values()))
            print(f"トータルマッチ率: {current_total_match_rate:.2%}")

            if current_total_match_rate > best_total_match_rate:
                best_total_match_rate = current_total_match_rate
                best_weights = self.penalty_weights.copy()
                print(f"*** 新しい最高のマッチ率を記録: {best_total_match_rate:.2%} ***")

            avg_match_rate = current_total_match_rate
            new_weights = self.penalty_weights.copy()
            for key in self.ordered_keys:
                if key in match_rates and match_rates[key] < avg_match_rate:
                    new_weights[key] *= learning_rate
            
            for j in range(len(self.ordered_keys) - 1):
                high_priority_key = self.ordered_keys[j]
                low_priority_key = self.ordered_keys[j+1]
                if new_weights[high_priority_key] <= new_weights[low_priority_key]:
                    new_weights[high_priority_key] = new_weights[low_priority_key] * 1.1

            self.penalty_weights = new_weights

        print("\n=== ペナルティ重みの自動最適化完了 ===")
        self.penalty_weights = best_weights
        print(f"最適化された重み: { {k: round(v, 2) for k, v in self.penalty_weights.items()} }")
        if best_total_match_rate != -1.0:
            print(f"最高のトータルマッチ率: {best_total_match_rate:.2%}")
            
    def calculate_penalty(self, student, partner):
        """学生とパートナー（リクルーター/社員）間のペナルティ計算"""
        penalty = 0
        for key in self.ordered_keys:
            student_val, partner_val = None, None
            if key == '地域':
                student_val, partner_val = student.get('地域'), partner.get('地域')
            elif self.partner_name_jp == 'リクルーター' and key == '所属':
                student_val, partner_val = student.get('学部名称'), partner.get('所属')
            else:
                student_val, partner_val = student.get(key), partner.get(key)
            
            if student_val is not None and partner_val is not None and student_val != partner_val:
                penalty += self.penalty_weights.get(key, 0)
        return penalty

    def solve_optimization(self, assignment_diff=10, verbose=True):
        """最適化問題の定式化と解法"""
        if self.partner_df is None:
            print("エラー: 先に set_partner() でマッチング対象を設定してください。")
            return None, None

        num_students = len(self.df_students)
        num_partners = len(self.partner_df)
        
        # config.pyの関数を呼び出して担当人数の範囲を計算
        min_assignments, max_assignments = config.calculate_assignment_range(
            num_students, num_partners, assignment_diff
        )

        if min_assignments is None:
            print(f"警告: {self.partner_name_jp}が0人のため、マッチングを実行できません。")
            return None, None
            
        if verbose:
            print(f"最適化問題の定式化...")
            print(f"変数数: {num_students} x {num_partners} = {num_students * num_partners}")

        prob = pl.LpProblem("Student_Partner_Matching", pl.LpMinimize)
        x = {(i, j): pl.LpVariable(f"x_{i}_{j}", cat='Binary') for i in range(num_students) for j in range(num_partners)}

        penalty_expr = pl.lpSum(self.calculate_penalty(self.df_students.iloc[i], self.partner_df.iloc[j]) * x[(i, j)]
                                for i in range(num_students) for j in range(num_partners))
        prob += penalty_expr, "Total_Penalty"

        for i in range(num_students):
            prob += pl.lpSum(x[(i, j)] for j in range(num_partners)) == 1

        for j in range(num_partners):
            prob += pl.lpSum(x[(i, j)] for i in range(num_students)) >= min_assignments
            prob += pl.lpSum(x[(i, j)] for i in range(num_students)) <= max_assignments
            
        if verbose:
            print(f"制約条件数: {len(prob.constraints)}")
            print("最適化を実行中...")

        prob.solve(pl.PULP_CBC_CMD(msg=0))
        status = pl.LpStatus[prob.status]
        
        if verbose:
            print(f"最適化ステータス: {status}")

        if status == 'Optimal':
            total_penalty = pl.value(prob.objective)
            if verbose:
                print(f"最適目的関数値（総ペナルティ）: {total_penalty}")

            matching_results = []
            partner_prefix = self.partner_name_jp
            for i in range(num_students):
                for j in range(num_partners):
                    if pl.value(x[(i, j)]) == 1:
                        student = self.df_students.iloc[i]
                        partner = self.partner_df.iloc[j]
                        penalty = self.calculate_penalty(student, partner)
                        
                        ### ▼▼▼ ここが修正箇所です ▼▼▼
                        result = {
                            # --- 学生情報 ---
                            '学生_応募者コード': student.get('応募者コード', 'N/A'),
                            '学生_カナ氏名': student.get('カナ氏名', 'N/A'),
                            '学生_大学名称': student.get('大学名称', 'N/A'),
                            '学生_学部名称': student.get('学部名称', 'N/A'),
                            '学生_学科名称': student.get('学科名称', 'N/A'),
                            '学生_応募者区分': student.get('応募者区分', 'N/A'),
                            '学生_文理区分': student.get('文理区分', 'N/A'),
                            '学生_性別': student.get('性別', 'N/A'),
                            '学生_性格タイプ': student.get('性格タイプ', 'N/A'),
                            '学生_地域': student.get('地域', 'N/A'),

                            # --- 担当者情報 ---
                            f'{partner_prefix}_社員番号': partner.get('社員番号', 'N/A'),
                            f'{partner_prefix}_カナ氏名': partner.get('カナ氏名', 'N/A'),
                            f'{partner_prefix}_大学名称': partner.get('大学名称', 'N/A'),
                            f'{partner_prefix}_学部名称': partner.get('学部名称', 'N/A'),
                            f'{partner_prefix}_学科名称': partner.get('学科名称', 'N/A'),
                            f'{partner_prefix}_応募者区分': partner.get('応募者区分', 'N/A'),
                            f'{partner_prefix}_文理区分': partner.get('文理区分', 'N/A'),
                            f'{partner_prefix}_性別': partner.get('性別', 'N/A'),
                            f'{partner_prefix}_性格タイプ': partner.get('性格タイプ', 'N/A'),
                            f'{partner_prefix}_地域': partner.get('地域', 'N/A'),
                            f'{partner_prefix}_所属': partner.get('所属', 'N/A'),

                            # --- マッチング情報 ---
                            'ペナルティ': penalty
                        }
                        ### ▲▲▲ 修正ここまで ▲▲▲
                        
                        matching_results.append(result)

            self.matching_results = pd.DataFrame(matching_results)
            
            if verbose:
                workload = self.matching_results[f'{partner_prefix}_社員番号'].value_counts().sort_index()
                print("\n=== 負荷分散結果 ===")
                for partner_id, count in workload.items():
                    print(f"{partner_id}: {count}人")
            return self.matching_results, total_penalty
        else:
            if verbose:
                print("最適解が見つかりませんでした。制約条件を見直してください。")
            return None, None

    def analyze_results(self):
        """結果の分析"""
        if not hasattr(self, 'matching_results') or self.matching_results is None:
            print("分析対象の結果がありません。")
            return None

        analysis_results = {}
        partner_prefix = self.partner_name_jp
        print("\n=== 属性一致度分析 ===")
        for key in self.ordered_keys:
            s_col, p_col, label = None, None, f"{key} 一致率"
            if key == '地域':
                s_col, p_col = '学生_地域', f'{partner_prefix}_地域'
            elif self.partner_name_jp == 'リクルーター' and key == '所属':
                s_col, p_col = '学生_学部名称', f'{partner_prefix}_所属'
                label = "所属一致率 (学生学部 vs 担当所属)"
            else:
                s_col, p_col = f'学生_{key}', f'{partner_prefix}_{key}'

            if s_col in self.matching_results.columns and p_col in self.matching_results.columns:
                match_rate = (self.matching_results[s_col] == self.matching_results[p_col]).mean()
                analysis_results[key] = match_rate
                print(f"{label}: {match_rate:.2%}")

        print(f"\n=== ペナルティ分析 ===")
        avg_penalty = self.matching_results['ペナルティ'].mean()
        min_penalty = self.matching_results['ペナルティ'].min()
        max_penalty = self.matching_results['ペナルティ'].max()
        total_penalty = self.matching_results['ペナルティ'].sum()
        analysis_results.update({'平均ペナルティ': avg_penalty, '最小ペナルティ': min_penalty,
                                 '最大ペナルティ': max_penalty, '総ペナルティ': total_penalty})
        print(f"平均ペナルティ: {avg_penalty:.2f}")
        print(f"最小ペナルティ: {min_penalty}")
        print(f"最大ペナルティ: {max_penalty}")
        print(f"総ペナルティ: {total_penalty}")
        
        return analysis_results
        
    def save_results(self, filename_prefix="matching_results"):
        """マッチング結果をCSVファイルとして保存"""
        if hasattr(self, 'matching_results') and self.matching_results is not None:
            partner_key = 'recruiters' if self.partner_name_jp == 'リクルーター' else 'employees'
            filename = f"{filename_prefix}_{partner_key}.csv"
            result_path = os.path.join(self.output_folder, filename)
            self.matching_results.to_csv(result_path, index=False, encoding='utf-8-sig')
            print(f"\nマッチング結果を {result_path} に保存しました。")
        else:
            print("保存する結果がありません。最適化を先に実行してください。")

    def save_optimized_weights(self, filename_prefix="optimized_weights"):
        """最適化されたペナルティ重みを大きい順にソートしてCSVファイルとして保存する"""
        if not self.penalty_weights:
            print("保存する重みがありません。重みの最適化を先に実行してください。")
            return
        
        # 重みを辞書からリストに変換し、値の大きい順にソート
        sorted_weights = sorted(self.penalty_weights.items(), key=lambda item: item[1], reverse=True)
        
        # DataFrameに変換
        df_weights = pd.DataFrame(sorted_weights, columns=['属性', '重み'])
        
        # 保存するファイル名を決定
        partner_key = 'recruiters' if self.partner_name_jp == 'リクルーター' else 'employees'
        filename = f"{filename_prefix}_{partner_key}.csv"
        result_path = os.path.join(self.output_folder, filename)
        
        # CSVファイルとして保存
        df_weights.to_csv(result_path, index=False, encoding='utf-8-sig')
        print(f"最適化された重みを {result_path} に保存しました。")

def main():
    """メイン関数"""
    print("=== 学生・担当社員マッチングシステム ===")
    print("ステップ2: 最適化問題の定式化と解法\n")

    optimizer = StudentRecruitingOptimizerFromCSV(
        student_csv='matching_data/students_data.csv',
        recruiter_csv='matching_data/recruiters_data.csv',
        employee_csv='matching_data/employees_data.csv',
        output_folder='output_results',
        seed=42
    )

    print("1. 全データの読み込みと前処理完了。")
    print(f"学生数: {len(optimizer.df_students)}")
    print(f"リクルーター数: {len(optimizer.df_recruiters_orig)}")
    print(f"現場代表社員数: {len(optimizer.df_employees_orig)}")

    # --- フェーズ1: 学生 vs リクルーターのマッチング ---
    optimizer.set_partner('recruiter')
    print("\n2-1. [リクルーター] ペナルティ重みの自動最適化を実行...")
    optimizer.optimize_penalty_weights(
        iterations=20, initial_base_weight=200, learning_rate=1.1,
        assignment_diff=const.ASSIGNMENT_DIFFERENCE
    )
    
    print("\n[リクルーター] 最適化された重みの保存...")
    optimizer.save_optimized_weights("optimized_weights")
    
    print("\n3-1. [リクルーター] 最適化された重みで最終的なマッチングを実行...")
    matching_results_rec, _ = optimizer.solve_optimization(
        assignment_diff=const.ASSIGNMENT_DIFFERENCE, verbose=True
    )

    if matching_results_rec is not None:
        print("\n4-1. [リクルーター] 最終結果の分析...")
        optimizer.analyze_results()
        print("\n5-1. [リクルーター] 結果の保存...")
        optimizer.save_results("matching_results")
    else:
        print("[リクルーター] マッチングの最適化に失敗しました。")

    # --- フェーズ2: 学生 vs 現場代表社員のマッチング ---
    optimizer.set_partner('employee')
    print("\n2-2. [現場代表社員] ペナルティ重みの自動最適化を実行...")
    optimizer.optimize_penalty_weights(
        iterations=20, initial_base_weight=200, learning_rate=1.1,
        assignment_diff=const.ASSIGNMENT_DIFFERENCE
    )

    print("\n[現場代表社員] 最適化された重みの保存...")
    optimizer.save_optimized_weights("optimized_weights")

    print("\n3-2. [現場代表社員] 最適化された重みで最終的なマッチングを実行...")
    matching_results_emp, _ = optimizer.solve_optimization(
        assignment_diff=const.ASSIGNMENT_DIFFERENCE, verbose=True
    )

    if matching_results_emp is not None:
        print("\n4-2. [現場代表社員] 最終結果の分析...")
        optimizer.analyze_results()
        print("\n5-2. [現場代表社員] 結果の保存...")
        optimizer.save_results("matching_results")
    else:
        print("[現場代表社員] マッチングの最適化に失敗しました。")
    
    print("\n=== 全ての最適化が完了しました ===")

if __name__ == "__main__":
    main()
