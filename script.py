import torch

# Load the original weights file
original_weights = torch.load('/home/xteam/zhaohao/pycharmproject/YTMT/merge_stem_reg_014_00055524.pt')

# Create a new weights dictionary
# new_weights = {}

# # Iterate through the original weights dictionary
# for key, value in original_weights.items():
#     # Check if the key contains 'projec_shit'
#     if 'projback_shit' in key:
#         # Replace 'projec_shit' with 'project_'
#         new_key = key.replace('projback_shit', 'projback_')
#         new_weights[new_key] = value
#     else:
#         # If the key doesn't contain 'projec_shit', keep it unchanged
#         new_weights[key] = value
#     if 'projback_shit_2' in key:
#         # Replace 'projec_shit' with 'project_'
#         new_key = key.replace('projback_shit_2', 'projback_2')
#         new_weights[new_key] = value
#     else:
#         # If the key doesn't contain 'projec_shit', keep it unchanged
#         new_weights[key] = value

# # Save the modified weights
# torch.save(new_weights, '/home/xteam/zhaohao/pycharmproject/RDNet/new_weights.pth')

# print("Weights file has been updated.")

# # 打印原始权重字典中的所有键,以检查确切的层名称
# print("原始权重文件中的层名:")
# for key in original_weights['icnn'].keys():
#     print(key)

# 创建一个新的权重字典
new_weights = {'icnn': {}}

# 遍历原始权重字典
for key, value in original_weights['icnn'].items():
    # 检查并替换包含 'projback_shit' 的键
    if 'projback_shit_2' in key:
        new_key = key.replace('projback_shit_2', 'projback_2')
        new_weights['icnn'][new_key] = value
    
    # 检查并替换包含 'projback_shit_2' 的键
    elif 'projback_shit' in key:
        new_key = key.replace('projback_shit', 'projback_')
        new_weights['icnn'][new_key] = value
    else:
        # 如果键不包含上述字符串,保持不变
        new_weights['icnn'][key] = value

# 打印新的权重字典中的所有键,以验证更改
print("\n更新后的权重文件中的层名:")
for key in new_weights['icnn'].keys():
    print(key)

# 保存修改后的权重
torch.save(new_weights, '/home/xteam/zhaohao/pycharmproject/RDNet/new_weights_4.pth')

print("\n权重文件已更新。")

