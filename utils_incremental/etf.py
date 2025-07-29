import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def generate_random_orthogonal_matrix(feat_in, num_classes):
    """生成随机正交矩阵"""
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)  # 使用QR分解生成正交矩阵
    orth_vec = torch.tensor(orth_vec).float()  # 转换为PyTorch张量
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "生成的矩阵不是正交矩阵"
    return orth_vec

def generate_etf_vector(in_channels, num_classes):
    """生成等距紧框架 (ETF) 向量"""
    # 生成正交矩阵
    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
    
    # 创建单位矩阵和全1矩阵
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    
    # 生成ETF向量
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    
    return etf_vec

# 测试脚本
if __name__ == "__main__":
    # 定义输入特征维度和类别数量
    in_channels = 512  # 输入特征维度，例如512
    num_classes = 200   # 类别数量，例如10类
    
    # 生成ETF向量
    etf_vec = generate_etf_vector(in_channels, num_classes)
    
    import pdb; pdb.set_trace()
    # 输出ETF向量的形状和部分内容
    print("ETF向量的形状: ", etf_vec.shape)
    print("ETF向量的前两列: \n", etf_vec[:, :2])
    
    # 验证ETF向量是否满足正交性
    orthogonality_check = torch.matmul(etf_vec.T, etf_vec)
    print("ETF向量的正交性检查 (应接近单位矩阵): \n", orthogonality_check)

    # Convert ETF vectors to NumPy array
    etf_vec_np = etf_vec.detach().numpy()
    np.save('etf_vec_1024.npy', etf_vec_np.T)
    # Apply TSNE to reduce the ETF vectors to 2D
    tsne = TSNE(n_components=2, random_state=42)
    etf_vec_2d = tsne.fit_transform(etf_vec_np.T)  # Transpose to get class-wise vectors

    # 2D TSNE Visualization
    fig, ax = plt.subplots()

    # Plot the 2D TSNE-transformed ETF vectors
    ax.scatter(etf_vec_2d[:, 0], etf_vec_2d[:, 1], color='blue')

    # Annotate points with class indices
    for i in range(num_classes):
        ax.text(etf_vec_2d[i, 0], etf_vec_2d[i, 1], f'Class {i}', fontsize=9)

    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    plt.title('2D TSNE Visualization of ETF Vectors')
    plt.savefig("cf100_2d_1024.png", dpi=300)
    # Apply TSNE to reduce the ETF vectors to 3D
    tsne = TSNE(n_components=3, random_state=42)  # Use 3 components for 3D visualization
    etf_vec_3d = tsne.fit_transform(etf_vec_np.T)  # Transpose to get class-wise vectors

    # 3D TSNE Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D TSNE-transformed ETF vectors
    ax.scatter(etf_vec_3d[:, 0], etf_vec_3d[:, 1], etf_vec_3d[:, 2], color='blue')

    # Annotate points with class indices
    # for i in range(num_classes):
    #     ax.text(etf_vec_3d[i, 0], etf_vec_3d[i, 1], etf_vec_3d[i, 2], f'Class {i}', fontsize=9)

    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_zlabel('TSNE Component 3')
    plt.title('3D TSNE Visualization of ETF Vectors')
    plt.savefig("cf100_3d_1024.png", dpi=300)


