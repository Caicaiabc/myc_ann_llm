import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_vector(s):
    """将字符串形式的向量解析为numpy数组"""
    return np.array(eval(s))


def find_closest_items(user_vector, item_vectors, item_ids, top_n=200):
    """找到与用户向量最接近的top_n个不重复的物品ID"""
    dot_products = np.array([np.dot(user_vector, item_vector) for item_vector in item_vectors])
    sorted_indices = np.argsort(dot_products)[::-1]

    unique_item_ids = []
    print(f"正在查找最接近的{top_n}个不重复物品......")
    for index in tqdm(sorted_indices, total=min(top_n, len(sorted_indices)), desc="进度"):
        item_id = item_ids[index]
        # 确保ID是字符串类型（根据实际数据调整）
        item_id = str(item_id)
        if item_id not in unique_item_ids:
            unique_item_ids.append(item_id)
        if len(unique_item_ids) == top_n:
            break

    if len(unique_item_ids) < top_n:
        unique_item_ids.extend([None] * (top_n - len(unique_item_ids)))

    return unique_item_ids


def check_target_presence(target_item_id, recommended_item_ids, positions):
    """检查目标物品是否在推荐列表的前k个位置中"""
    # 处理target_item_id，如果是字符串形式的列表则解析
    if isinstance(target_item_id, str) and (target_item_id.startswith('[') and target_item_id.endswith(']')):
        try:
            target_item_id_list = eval(target_item_id)
            if len(target_item_id_list) > 0:
                target_item_id = str(target_item_id_list[0])
            else:
                return {f"in_top_{pos}": False for pos in positions}
        except:
            target_item_id = str(target_item_id)
    else:
        target_item_id = str(target_item_id)

    presence = {}
    for pos in positions:
        presence[f"in_top_{pos}"] = target_item_id in recommended_item_ids[:pos]
    return presence


# 读取文件
item_df = pd.read_csv('item_res.csv')
user_df = pd.read_csv('user_res.csv')

# 解析向量列
item_df['item_res'] = item_df['item_res'].apply(parse_vector)
user_df['user_res'] = user_df['user_res'].apply(parse_vector)

# 准备数据
item_vectors = item_df['item_res'].tolist()
# 确保物品ID是字符串类型
item_ids = [str(id) for id in item_df['item_id'].tolist()]
user_vectors = user_df['user_res'].tolist()
sample_ids = user_df['sample_id'].tolist()
# 确保目标物品ID是字符串类型
target_item_ids = [str(id) for id in user_df['target_item_item_id'].tolist()]

# 要检查的位置
positions = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]

results = []
for i in range(len(user_vectors)):
    user_vector = user_vectors[i]
    sample_id = sample_ids[i]
    target_item_id = target_item_ids[i]

    recommended_item_ids = find_closest_items(user_vector, item_vectors, item_ids)
    presence = check_target_presence(target_item_id, recommended_item_ids, positions)

    result_row = {
        'sample_id': sample_id,
        'target_item_item_id': target_item_id,
        'tuijian_res': recommended_item_ids,
        **presence
    }
    results.append(result_row)

# 将结果转换为DataFrame并保存为CSV文件
final_res_df = pd.DataFrame(results)
final_res_df.to_csv('final_res.csv', index=False)