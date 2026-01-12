"""
线性回归模型示例
目标：用一条直线来拟合数据点，找到最佳的直线方程 y = w*x + b

举个例子：
假设我们有一些数据点，比如房子面积和价格的关系
面积(x): 50, 100, 150, 200 平方米
价格(y): 150, 300, 450, 600 万元
我们要找到一个公式，能根据面积预测价格
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统使用Arial Unicode MS
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可以用这个
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子，让每次运行结果一样（方便调试）
torch.manual_seed(42)


# 第一步：准备数据
# 我们生成一些假数据，真实规律是 y = 2*x + 3（加上一些随机噪声）
def create_data():
    """
    创建训练数据
    返回：
        x: 输入数据（比如房子面积）
        y: 输出数据（比如房子价格）
    """
    # 生成100个从0到10的随机数作为输入
    x = torch.randn(100, 1) * 10
    
    # 真实的关系：y = 2*x + 3，再加上一些随机噪声
    y = 2 * x + 3 + torch.randn(100, 1) * 2
    
    return x, y


# 第二步：定义模型
# 这就是我们要训练的"直线公式"
class LinearRegressionModel(nn.Module):
    """
    线性回归模型
    模型公式：y = w*x + b
    其中：
        w (weight): 权重，相当于直线的斜率
        b (bias): 偏置，相当于直线的截距
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层：输入是1个特征，输出是1个值
        # 这个层内部包含了 w 和 b 两个参数
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        """
        前向传播：计算预测值
        输入 x，输出预测的 y
        """
        return self.linear(x)


# 第三步：训练模型
def train_model():
    """
    训练线性回归模型的完整流程
    """
    # 1. 准备数据
    x_train, y_train = create_data()
    print(f"训练数据数量：{len(x_train)} 个样本")
    
    # 2. 创建模型
    model = LinearRegressionModel()
    print(f"\n初始模型参数：")
    print(f"  w (权重/斜率) = {model.linear.weight.item():.4f}")
    print(f"  b (偏置/截距) = {model.linear.bias.item():.4f}")
    
    # 3. 定义损失函数
    # MSE (Mean Squared Error): 均方误差，用来衡量预测值和真实值的差距
    # 差距越小，说明模型越好
    criterion = nn.MSELoss()
    
    # 4. 定义优化器
    # SGD (Stochastic Gradient Descent): 随机梯度下降
    # lr (learning rate): 学习率，控制每次调整参数的步长
    # 学习率太大可能错过最优解，太小则训练太慢
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 5. 开始训练
    epochs = 100  # 训练100轮
    losses = []   # 记录每轮的损失值
    
    print(f"\n开始训练，共 {epochs} 轮...")
    for epoch in range(epochs):
        # (1) 前向传播：用当前模型预测
        y_pred = model(x_train)
        
        # (2) 计算损失：预测值和真实值的差距
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        
        # (3) 反向传播：计算梯度（每个参数该如何调整）
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()         # 计算新的梯度
        
        # (4) 更新参数：根据梯度调整 w 和 b
        optimizer.step()
        
        # 每10轮打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"第 {epoch+1:3d} 轮 - 损失值: {loss.item():.4f}")
    
    # 6. 训练完成，查看最终参数
    print(f"\n训练完成！")
    print(f"最终模型参数：")
    print(f"  w (权重/斜率) = {model.linear.weight.item():.4f}")
    print(f"  b (偏置/截距) = {model.linear.bias.item():.4f}")
    print(f"  （真实值应该接近: w=2, b=3）")
    
    # 7. 可视化结果
    visualize_results(model, x_train, y_train, losses)
    
    return model


def visualize_results(model, x_train, y_train, losses):
    """
    可视化训练结果
    """
    # 创建一个包含2个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：数据点和拟合直线
    with torch.no_grad():  # 不需要计算梯度
        # 生成一条直线用于显示
        x_line = torch.linspace(x_train.min(), x_train.max(), 100).reshape(-1, 1)
        y_line = model(x_line)
    
    ax1.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label='训练数据')
    ax1.plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, label='拟合直线')
    ax1.set_xlabel('x (输入)', fontsize=12)
    ax1.set_ylabel('y (输出)', fontsize=12)
    ax1.set_title('线性回归拟合结果', fontsize=14)
    ax1.legend(prop={'size': 11})
    ax1.grid(True, alpha=0.3)
    
    # 右图：训练过程中的损失值变化
    ax2.plot(losses, 'b-', linewidth=1)
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('损失值 (Loss)', fontsize=12)
    ax2.set_title('训练过程 - 损失值变化', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_result.png', dpi=100, bbox_inches='tight')
    print(f"\n可视化结果已保存到: linear_regression_result.png")
    plt.show()


def test_model(model):
    """
    测试模型：用训练好的模型进行预测
    """
    print(f"\n{'='*50}")
    print(f"模型测试 - 进行一些预测")
    print(f"{'='*50}")
    
    # 测试几个输入值
    test_inputs = [1.0, 5.0, 10.0]
    
    with torch.no_grad():  # 测试时不需要计算梯度
        for x_val in test_inputs:
            x_test = torch.tensor([[x_val]])
            y_pred = model(x_test)
            y_true = 2 * x_val + 3  # 真实值
            print(f"输入 x = {x_val:5.1f}  →  预测 y = {y_pred.item():6.2f}  (真实值约为 {y_true:.2f})")


# 主程序
if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch 线性回归模型示例")
    print("=" * 50)
    
    # 训练模型
    model = train_model()
    
    # 测试模型
    test_model(model)
    
    print(f"\n{'='*50}")
    print("程序执行完成！")
    print("=" * 50)
