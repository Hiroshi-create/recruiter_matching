

# 担当人数の割り当て許容差
# この値を大きくすると、各担当者に割り当てられる学生数の差が大きくなる可能性があります。

def calculate_assignment_range(num_students: int, num_partners: int, assignment_diff: int) -> tuple[int, int] | tuple[None, None]:
    """
    担当者一人当たりの担当学生数の下限と上限を計算します。

    Args:
        num_students (int): 学生の総数。
        num_partners (int): 担当者の総数。
        assignment_diff (int): 担当人数の許容差。

    Returns:
        tuple[int, int] | tuple[None, None]: (最小担当人数, 最大担当人数) のタプルを返します。
                                             担当者が0人の場合は (None, None) を返します。
    """
    if num_partners <= 0:
        return None, None

    avg_assignments = num_students / num_partners
    min_assignments = max(1, int(round(avg_assignments - assignment_diff / 2)))
    max_assignments = int(round(avg_assignments + assignment_diff / 2))

    # 最小値が最大値を上回る稀なケースに対応
    if max_assignments < min_assignments:
        max_assignments = min_assignments
        
    return min_assignments, max_assignments
