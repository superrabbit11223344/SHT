import pickle
import numpy as np

TRAIN_PATH = 'Data/gowalla/trnMat.pkl'
TEST_PATH = 'Data/gowalla/tstMat.pkl'

# 读取训练集
with open(TRAIN_PATH, 'rb') as f:
    trnMat = pickle.load(f)

# 读取测试集
with open(TEST_PATH, 'rb') as f:
    tstMat = pickle.load(f)

print(tstMat.data)
print(type(tstMat))

# 训练集中的user数和item数
trnUsers = set()
trnItems = set()
trnInters = set()
for i in range(len(trnMat.data)):
    row = trnMat.row[i]
    col = trnMat.col[i]
    trnUsers.add(row)
    trnItems.add(col)
    trnInters.add((row, col))

print(f'训练集交互用户数：{len(trnUsers)}')
print(f'训练集交互商品数：{len(trnItems)}')
print(f'交互记录数：{len(trnMat.data)}')
print(f'交互矩阵shape: {trnMat.shape}')
print()

# 测试集中的user数和item数
tstUsers = set()
tstItems = set()
tstInters = set()
for i in range(len(tstMat.data)):
    row = tstMat.row[i]
    col = tstMat.col[i]
    tstUsers.add(row)
    tstItems.add(col)
    tstInters.add((row, col))

print(f'测试集交互用户数: {len(tstUsers)}')
print(f'测试集交互商品数: {len(tstItems)}')
print(f'交互记录数: {len(tstMat.data)}')
print(f'交互矩阵shape: {tstMat.shape}')

# 查看训练集中的交互记录trnInters, 和测试集中的交互记录tstInters是否有交集
print('训练集中的交互记录trnInters, 和测试集中的交互记录tstInters是否有交集: ', trnInters & tstInters == set())
print()

# user id范围，进行了去重和排序操作
print('训练集user id范围: ', np.unique(np.sort(trnMat.row)))
print('测试集user id范围: ', np.unique(np.sort(tstMat.row)))

# item id范围，进行了去重和排序操作
print('训练集item id范围: ', np.unique(np.sort(trnMat.col)))
print('测试集item id范围: ', np.unique(np.sort(tstMat.col)))
