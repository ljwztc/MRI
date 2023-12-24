import torch

# 定义两个二维矩阵
matrix1 = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

matrix2 = torch.Tensor([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0]
            ])

# 将矩阵转换为四维张量（批次维度为1）
matrix1 = matrix1.unsqueeze(0).unsqueeze(0).float()
matrix2 = matrix2.unsqueeze(0).unsqueeze(0).float()

# 使用PyTorch的卷积函数计算卷积结果
conv = torch.nn.Conv2d(1, 1, kernel_size=matrix2.shape[2:], bias=False)
conv.weight.data = matrix2
conv_result = conv(matrix1)

print(conv_result.squeeze().int())