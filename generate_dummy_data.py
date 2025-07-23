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
- 生成する社員・リクルーターの人数に基づき、使用する大学を動的に選定します。
- 全てのダミーデータ（学生、社員、リクルーター）は、その選定された大学リストのみを使用して生成されます。
- 社員とリクルーターには、選定された大学が網羅的に割り当てられます。

必要なライブラリ:
pip install pandas numpy
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any
import os
import data_definitions as const

class MatchingDataGenerator:
    """
    学生、現場代表社員、リクルーターのダミーデータを新しいスキーマで生成するクラス
    """

    def __init__(self, seed: int = 42, output_folder: str = "matching_data"):
        """
        初期化

        Args:
            seed: ランダムシードの設定
            output_folder: 出力先フォルダ名
        """
        self.seed = seed
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
        
        # --- 今回の生成で使用する大学のリスト ---
        self._selected_universities: List[str] = []

    def _generate_kana_name(self, gender: str) -> str:
        surname = random.choice(self._surnames)
        if gender == '男性':
            given_name = random.choice(self._male_names)
        else:
            given_name = random.choice(self._female_names)
        return f"{surname}　{given_name}"

    def _generate_education_info(self, university: str) -> Dict[str, str]:
        """
        指定された大学名に基づいて学歴情報を生成する。所在地情報も追加で返す。
        
        Args:
            university: 使用する大学名

        Returns:
            学歴情報を含む辞書
        """
        uni = university
        
        # 有効な学部を持つ大学・文理区分が見つかるまでループ
        while True:
            # その大学に存在する、空ではない学部リストを持つ文理区分のリストを作成
            valid_divisions = [
                division for division, faculties in self._university_map[uni].items() if faculties
            ]
            
            # 有効な文理区分が存在する場合のみループを抜ける
            if valid_divisions:
                break
            else:
                # 万が一、指定された大学に有効な学部情報がない場合、別の大学で試す
                print(f"警告: 大学 '{uni}' に有効な学部情報がありません。別の大学をランダムに選択します。")
                uni = random.choice(list(self._university_map.keys()))

        # 確定した大学・文理区分から学部・学科を選択
        division = random.choice(valid_divisions)
        faculty_dict = self._university_map[uni][division]
        faculty = random.choice(list(faculty_dict.keys()))
        department = random.choice(faculty_dict[faculty])
        
        # 確定した大学名から所在地を取得
        location = self._university_location_map.get(uni, random.choice(self._work_locations))
        
        return {
            '大学名称': uni,
            '文理区分': division,
            '学部名称': faculty,
            '学科名称': department,
            '所在地': location
        }

    def generate_students(self, num_students: int = 500) -> pd.DataFrame:
        students = []
        for i in range(num_students):
            gender = random.choice(self._genders)
            # 選定された大学リストからランダムに大学を選択
            uni_name = random.choice(self._selected_universities)
            education_info = self._generate_education_info(uni_name)
            
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
        # 選定された大学リストをシャッフルし、割り当て準備
        universities_to_assign = self._selected_universities.copy()
        random.shuffle(universities_to_assign)

        for i in range(num_employees):
            gender = random.choice(self._genders)
            
            # 選定大学を循環的に割り当て、すべての大学が使われるようにする
            uni_name = universities_to_assign[i % len(universities_to_assign)]
            education_info = self._generate_education_info(uni_name)

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
        # 選定された大学リストをシャッフルし、割り当て準備
        universities_to_assign = self._selected_universities.copy()
        random.shuffle(universities_to_assign)

        for i in range(num_recruiters):
            gender = random.choice(self._genders)

            # 選定大学を循環的に割り当て、すべての大学が使われるようにする
            uni_name = universities_to_assign[i % len(universities_to_assign)]
            education_info = self._generate_education_info(uni_name)

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
        生成前に、使用する大学のリストを選定する。
        """
        # --- Step 1: 使用する大学を選定 ---
        num_universities_to_select = min(num_employees, num_recruiters)
        all_universities = list(self._university_map.keys())
        
        if num_universities_to_select > len(all_universities):
            print(f"警告: 選択する大学数({num_universities_to_select})が定義済みの大学総数({len(all_universities)})を超えています。")
            print("利用可能な全ての大学を使用します。")
            self._selected_universities = all_universities
        else:
            self._selected_universities = random.sample(all_universities, k=num_universities_to_select)

        print(f"\n--- 今回のデータ生成で使用する大学 ({len(self._selected_universities)}校) ---")
        # 10校まで表示
        for uni in self._selected_universities[:10]:
            print(f"- {uni}")
        if len(self._selected_universities) > 10:
            print(f"...他{len(self._selected_universities) - 10}校")
        print("-------------------------------------------\n")
        
        # --- Step 2: 選定された大学リストに基づいて各データを生成 ---
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
                # value_counts()が空でないか確認
                if not df[col].empty:
                    print(df[col].value_counts().head())
                else:
                    print("データがありません。")

def main():
    print("=== 学生・現場代表社員・リクルーターマッチングシステム ===")
    print("Step 1: ダミーデータ生成\n")

    generator = MatchingDataGenerator(seed=42, output_folder="matching_data")

    print("新しいスキーマに基づいてダミーデータを生成しています...")
    # generate_all_data内で大学の選定とデータ生成が行われる
    data_dict = generator.generate_all_data(
        num_students=50,
        num_employees=5,
        num_recruiters=5
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
