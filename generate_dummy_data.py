#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生・現場代表社員・リクルーターのマッチングシステム
Step 1: ダミーデータ生成 (改訂版)

このスクリプトは、指定された新しいスキーマに基づいて、最適化問題用のダミーデータを生成します。
- 学生データ: 応募者情報、学歴、性格タイプ、合否など
- 現場代表社員データ: 社員情報、学歴、勤務地、性格タイプなど
- リクルーターデータ: 社員情報、学歴、勤務地、所属など

[主な変更点]
- **実行のたびにランダムな結果が得られるように、乱数シードの固定を解除しました。**
  (結果を固定したい場合は、`MatchingDataGenerator(seed=任意の値)`でシードを指定してください)
- 「関西学院大学」が必ず選定リストに含まれるロジックは維持しています。
- 生成する社員・リクルーターの人数に基づき、使用する「大学と学部のペア」を動的に選定します。
- 全てのダミーデータは、その選定されたペアのみを使用して生成されます。

必要なライブラリ:
pip install pandas numpy
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any, Optional
import os
import data_definitions as const

class MatchingDataGenerator:
    """
    学生、現場代表社員、リクルーターのダミーデータを新しいスキーマで生成するクラス
    """

    def __init__(self, seed: Optional[int] = None, output_folder: str = "matching_data"):
        """
        初期化

        Args:
            seed: ランダムシードの設定。Noneの場合、実行のたびに結果が変わります。
            output_folder: 出力先フォルダ名
        """
        self.seed = seed
        # seedが指定されている場合のみ乱数を固定する
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # --- データ定義 ---
        self._surnames = const.SURNAMES
        self._male_names = const.MALE_NAMES
        self._female_names = const.FEMALE_NAMES
        self._university_location_map = const.UNIVERSITY_LOCATION_MAP
        self._university_map = const.UNIVERSITY_MAP
        self._genders = const.GENDERS
        self._applicant_categories = const.APPLICANT_CATEGORIES
        self._personality_types = const.PERSONALITY_TYPES
        self._results = const.RESULTS
        self._work_locations = const.WORK_LOCATIONS
        self._corporate_departments = const.CORPORATE_DEPARTMENTS
        
        # --- 今回の生成で使用する大学と学部のペアリスト ---
        self._selected_uni_faculty_pairs: List[Dict[str, str]] = []

    def _generate_kana_name(self, gender: str) -> str:
        surname = random.choice(self._surnames)
        if gender == '男性':
            given_name = random.choice(self._male_names)
        else:
            given_name = random.choice(self._female_names)
        return f"{surname}　{given_name}"

    def _generate_education_info(self, university: str, division: str, faculty: str) -> Dict[str, str]:
        """
        指定された大学・文理区分・学部名に基づいて学歴情報を生成する。
        
        Args:
            university: 使用する大学名
            division: 使用する文理区分
            faculty: 使用する学部名

        Returns:
            学歴情報を含む辞書
        """
        department = random.choice(self._university_map[university][division][faculty])
        location = self._university_location_map.get(university, random.choice(self._work_locations))
        
        return {
            '大学名称': university,
            '文理区分': division,
            '学部名称': faculty,
            '学科名称': department,
            '所在地': location
        }

    def generate_students(self, num_students: int = 500) -> pd.DataFrame:
        students = []
        if not self._selected_uni_faculty_pairs:
            print("警告: 学生データ生成に使用できる大学・学部ペアがありません。空のDataFrameを返します。")
            return pd.DataFrame()
            
        for i in range(num_students):
            gender = random.choice(self._genders)
            
            selected_pair = random.choice(self._selected_uni_faculty_pairs)
            education_info = self._generate_education_info(
                selected_pair['university'],
                selected_pair['division'],
                selected_pair['faculty']
            )
            
            student_record = {
                '応募者コード': f'X{10000000 + i}',
                'カナ氏名': self._generate_kana_name(gender),
                '性別': gender,
                '大学名称': education_info['大学名称'],
                '学部名称': education_info['学部名称'],
                '学科名称': education_info['学科名称'],
                '応募者区分': random.choice(self._applicant_categories),
                '文理区分': education_info['文理区分'],
                '性格タイプ': random.choice(self._personality_types),
                '合否': random.choice(self._results),
                '東or西': education_info['所在地']
            }
            students.append(student_record)
        
        columns_order = ['応募者コード', 'カナ氏名', '性別', '大学名称', '学部名称', '学科名称', '応募者区分', '文理区分', '性格タイプ', '合否', '東or西']
        return pd.DataFrame(students, columns=columns_order)

    def generate_employees(self, num_employees: int = 40) -> pd.DataFrame:
        employees = []
        if not self._selected_uni_faculty_pairs:
            print("警告: 社員データ生成に使用できる大学・学部ペアがありません。空のDataFrameを返します。")
            return pd.DataFrame()

        pairs_to_assign = self._selected_uni_faculty_pairs.copy()
        random.shuffle(pairs_to_assign)

        for i in range(num_employees):
            gender = random.choice(self._genders)
            
            selected_pair = pairs_to_assign[i % len(pairs_to_assign)]
            education_info = self._generate_education_info(
                selected_pair['university'],
                selected_pair['division'],
                selected_pair['faculty']
            )

            employee_record = {
                '社員番号': 1234500 + i,
                'カナ氏名': self._generate_kana_name(gender),
                '大学名称': education_info['大学名称'],
                '学部名称': education_info['学部名称'],
                '学科名称': education_info['学科名称'],
                '応募者区分': random.choice(self._applicant_categories),
                '文理区分': education_info['文理区分'],
                '性別': gender,
                '勤務地': random.choice(self._work_locations),
                '性格タイプ': random.choice(self._personality_types)
            }
            employees.append(employee_record)
            
        columns_order = ['社員番号', 'カナ氏名', '大学名称', '学部名称', '学科名称', '応募者区分', '文理区分', '性別', '勤務地', '性格タイプ']
        return pd.DataFrame(employees, columns=columns_order)

    def generate_recruiters(self, num_recruiters: int = 40) -> pd.DataFrame:
        recruiters = []
        if not self._selected_uni_faculty_pairs:
            print("警告: リクルーターデータ生成に使用できる大学・学部ペアがありません。空のDataFrameを返します。")
            return pd.DataFrame()
            
        pairs_to_assign = self._selected_uni_faculty_pairs.copy()
        random.shuffle(pairs_to_assign)

        for i in range(num_recruiters):
            gender = random.choice(self._genders)

            selected_pair = pairs_to_assign[i % len(pairs_to_assign)]
            education_info = self._generate_education_info(
                selected_pair['university'],
                selected_pair['division'],
                selected_pair['faculty']
            )

            recruiter_record = {
                '社員番号': 7654300 + i,
                'カナ氏名': self._generate_kana_name(gender),
                '大学名称': education_info['大学名称'],
                '学部名称': education_info['学部名称'],
                '学科名称': education_info['学科名称'],
                '応募者区分': random.choice(self._applicant_categories),
                '文理区分': education_info['文理区分'],
                '性別': gender,
                '勤務地': random.choice(self._work_locations),
                '所属': random.choice(self._corporate_departments)
            }
            recruiters.append(recruiter_record)
        
        columns_order = ['社員番号', 'カナ氏名', '大学名称', '学部名称', '学科名称', '応募者区分', '文理区分', '性別', '勤務地', '所属']
        return pd.DataFrame(recruiters, columns=columns_order)

    def generate_all_data(self, num_students: int, num_employees: int, num_recruiters: int) -> Dict[str, pd.DataFrame]:
        """
        全てのダミーデータを生成する。
        生成前に、使用する「大学と学部のペア」リストを選定する
        - 関西学院大学/理系/工学部ペアは必ず含める
        - 東京大学, 慶應義塾大学, 同志社大学, 青山学院大学については大学名のみ一致でペアを必ず含める（文理・学部はランダム）
        """
        required_universities = ['東京大学', '慶應義塾大学', '同志社大学', '青山学院大学']
        fixed_university = "関西学院大学"
        fixed_division   = "理系"
        fixed_faculty    = "工学部"
        self._selected_uni_faculty_pairs = []

        # 1. 全ペア走査で記録
        all_uni_faculty_pairs = []
        fixed_pair = None
        candidate_pairs_for_univ = {name: [] for name in required_universities}
        for uni, divisions in self._university_map.items():
            for division, faculties in divisions.items():
                if faculties:
                    for faculty, departments in faculties.items():
                        pair = {'university': uni, 'division': division, 'faculty': faculty}
                        all_uni_faculty_pairs.append(pair)
                        # まず関学/理系/工学部ペア
                        if uni == fixed_university and division == fixed_division and faculty == fixed_faculty:
                            fixed_pair = {'university': uni, 'division': division, 'faculty': faculty}
                        # 東京大学, 慶應義塾大学, 同志社大学, 青山学院大学は候補として記憶
                        if uni in required_universities:
                            candidate_pairs_for_univ[uni].append({'university': uni, 'division': division, 'faculty': faculty})
        num_pairs_to_select = min(num_employees, num_recruiters)

        # 2. 関学/理系/工学部は必ず
        if fixed_pair is not None:
            self._selected_uni_faculty_pairs.append(fixed_pair)
        else:
            print(f"警告: {fixed_university} / {fixed_division} / {fixed_faculty} がデータ定義にありません！")

        # 3. 各大学名ごとに「ランダムなペア」を抽出＆追加（重複回避）
        for uni in required_universities:
            cand = candidate_pairs_for_univ[uni]
            if not cand:
                print(f"警告: {uni} の有効な学部ペアが見つかりません！")
                continue
            # 重複回避
            used = set((p['university'], p['division'], p['faculty']) for p in self._selected_uni_faculty_pairs)
            remain = [p for p in cand if (p['university'], p['division'], p['faculty']) not in used]
            if not remain:
                print(f"警告: {uni} ですでにペアが採用されているか、候補がありません！")
                continue
            pick = random.choice(remain)
            self._selected_uni_faculty_pairs.append(pick)

        # 4. 残り枠を全候補から補充（すでに使ったペアは除く）
        already = set((p['university'], p['division'], p['faculty']) for p in self._selected_uni_faculty_pairs)
        candidates = [p for p in all_uni_faculty_pairs
                    if (p['university'], p['division'], p['faculty']) not in already]
        rest = num_pairs_to_select - len(self._selected_uni_faculty_pairs)
        if rest > 0 and len(candidates) > 0:
            self._selected_uni_faculty_pairs.extend(random.sample(candidates, k=min(rest, len(candidates))))

        random.shuffle(self._selected_uni_faculty_pairs)

        print(f"\n--- 今回のデータ生成で使用する大学-文理-学部ペア ({len(self._selected_uni_faculty_pairs)}組) ---")
        for pair in self._selected_uni_faculty_pairs[:10]:
            print(f"- {pair['university']} / {pair['division']} / {pair['faculty']}")
        if len(self._selected_uni_faculty_pairs) > 10:
            print(f"...他{len(self._selected_uni_faculty_pairs) - 10}組")
        print("--------------------------------------------------\n")

        return {
            'students': self.generate_students(num_students),
            'employees': self.generate_employees(num_employees),
            'recruiters': self.generate_recruiters(num_recruiters)
        }

    def save_data(self, data: Dict[str, pd.DataFrame]) -> None:
        for key, df in data.items():
            filename = f"{key}_data.csv"
            filepath = os.path.join(self.output_folder, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"保存完了: {filepath}")

    def display_statistics(self, data: Dict[str, pd.DataFrame]) -> None:
        id_columns = ['応募者コード', '社員番号', 'カナ氏名']
        for key, df in data.items():
            print(f"\n=== {key.upper()} データの統計情報 (上位5件) ===")
            print(f"総件数: {len(df)}件")
            for col in df.columns:
                if col in id_columns:
                    continue
                print(f"\n--- {col} ---")
                if not df.empty and col in df.columns and not df[col].empty:
                    print(df[col].value_counts().head())
                else:
                    print("データがありません。")

def main():
    print("=== 学生・現場代表社員・リクルーターマッチングシステム ===")
    print("Step 1: ダミーデータ生成\n")

    # 毎回異なるデータを生成するため、seedは指定しません。
    # 特定の結果を再現したい場合は、seed=42 のようにシード値を指定してください。
    generator = MatchingDataGenerator(output_folder="matching_data")

    print("新しいスキーマに基づいてダミーデータを生成しています...")
    data_dict = generator.generate_all_data(
        num_students=const.NUM_STUDENTS,
        num_employees=const.NUM_EMPLOYEES,
        num_recruiters=const.NUM_RECRUITERS
    )

    generator.display_statistics(data_dict)

    print("\n\n=== データ保存 ===")
    generator.save_data(data_dict)

    print("\n\n=== データ生成完了 ===")
    print("以下のファイルが `matching_data` フォルダに生成されました:")
    print("- students_data.csv")
    print("- employees_data.csv")
    print("- recruiters_data.csv")
    print("\n次のステップでは、これらのデータを使用して最適化計算を実行します。")

if __name__ == "__main__":
    main()
