"""
逻辑回归模型示例
目标：根据特征判断属于哪一类（二分类问题）

举个例子：
假设我们要判断一个学生是否能通过考试
输入(x): 学习时长（小时）
输出(y): 0=不及格, 1=及格

与线性回归的区别：
- 线性回归：预测连续的数值（比如预测价格是150、200、300...)
- 逻辑回归：预测分类（比如预测及格/不及格，也就是0或1）
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统使用Arial Unicode MS
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可以用这个
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子，让每次运行结果一样（方便调试）
torch.manual_seed(42)


# 第一步：准备数据
# 生成分类数据：学习时长少的学生不及格(0)，学习时长多的学生及格(1)
def create_data():
    """
    创建二分类训练数据
    返回：
        x: 输入特征（比如学习时长）
        y: 分类标签（0或1）
    """
    # 生成200个样本
    n_samples = 200
    
    # 第一类：学习时长较短的学生（不及格，标签=0）
    # 学习时长在 2-5 小时之间
    x_class0 = torch.randn(n_samples // 2, 1) * 1.5 + 3.5
    y_class0 = torch.zeros(n_samples // 2, 1)  # 标签都是0
    
    # 第二类：学习时长较长的学生（及格，标签=1）
    # 学习时长在 5-8 小时之间
    x_class1 = torch.randn(n_samples // 2, 1) * 1.5 + 6.5
    y_class1 = torch.ones(n_samples // 2, 1)   # 标签都是1
    
    # 合并两类数据
    x = torch.cat([x_class0, x_class1], dim=0)
    y = torch.cat([y_class0, y_class1], dim=0)
    
    # 打乱数据顺序
    indices = torch.randperm(n_samples)
    x = x[indices]
    y = y[indices]
    
    return x, y


# 第二步：定义模型
# 逻辑回归 = 线性变换 + Sigmoid激活函数
class LogisticRegressionModel(nn.Module):
    """
    逻辑回归模型
    
    公式分两步：
    1. 线性变换：z = w*x + b
    2. Sigmoid激活：y = 1 / (1 + e^(-z))
    
    Sigmoid函数的作用：
    - 把任意实数压缩到 0-1 之间
    - 输出可以理解为"属于类别1的概率"
    - 如果 y > 0.5，预测为类别1
    - 如果 y < 0.5，预测为类别0
    """
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # 线性层：输入1个特征，输出1个值
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        """
        前向传播：计算预测值
        
        具体计算过程（假设 x=5, w=2, b=-8）：
        1. 线性计算：z = w*x + b = 2*5 + (-8) = 2
        2. Sigmoid计算：y = 1/(1+e^(-2)) = 1/(1+0.135) ≈ 0.88
        3. 判断分类：0.88 > 0.5，所以预测为类别1（及格）
        """
        z = self.linear(x)           # 步骤1：线性变换
        y = torch.sigmoid(z)         # 步骤2：Sigmoid激活
        return y


# 第三步：训练模型
def train_model():
    """
    训练逻辑回归模型的完整流程
    """
    # 1. 准备数据
    x_train, y_train = create_data()
    print(f"训练数据数量：{len(x_train)} 个样本")
    print(f"  其中类别0（不及格）: {(y_train == 0).sum().item()} 个")
    print(f"  其中类别1（及格）: {(y_train == 1).sum().item()} 个")
    
    # 2. 创建模型
    model = LogisticRegressionModel()
    print(f"\n初始模型参数：")
    print(f"  w (权重) = {model.linear.weight.item():.4f}")
    print(f"  b (偏置) = {model.linear.bias.item():.4f}")
    
    # 3. 定义损失函数
    # BCELoss (Binary Cross Entropy Loss): 二元交叉熵损失
    # 专门用于二分类问题，衡量预测概率和真实标签的差距
    criterion = nn.BCELoss()
    
    # 4. 定义优化器
    # 使用 SGD (随机梯度下降)
    # 学习率设为 0.1（比线性回归大一些，因为逻辑回归收敛较慢）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # 5. 开始训练
    epochs = 200  # 训练200轮（比线性回归多，因为分类问题更复杂）
    losses = []   # 记录每轮的损失值
    accuracies = []  # 记录每轮的准确率
    
    print(f"\n开始训练，共 {epochs} 轮...")
    for epoch in range(epochs):
        # (1) 前向传播：用当前模型预测
        y_pred = model(x_train)
        
        # (2) 计算损失：预测概率和真实标签的差距
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        
        # (3) 计算准确率：看预测对了多少
        # 如果预测概率 > 0.5，判定为类别1；否则为类别0
        predictions = (y_pred > 0.5).float()
        accuracy = (predictions == y_train).float().mean()
        accuracies.append(accuracy.item())
        
        # (4) 反向传播：计算梯度
        optimizer.zero_grad()
        loss.backward()
        
        # (5) 更新参数
        optimizer.step()
        
        # 每20轮打印一次进度
        if (epoch + 1) % 20 == 0:
            print(f"第 {epoch+1:3d} 轮 - 损失值: {loss.item():.4f}, 准确率: {accuracy.item()*100:.2f}%")
    
    # 6. 训练完成，查看最终参数
    print(f"\n训练完成！")
    print(f"最终模型参数：")
    print(f"  w (权重) = {model.linear.weight.item():.4f}")
    print(f"  b (偏置) = {model.linear.bias.item():.4f}")
    print(f"最终准确率: {accuracies[-1]*100:.2f}%")
    
    # 7. 可视化结果
    visualize_results(model, x_train, y_train, losses, accuracies)
    
    return model


def visualize_results(model, x_train, y_train, losses, accuracies):
    """
    可视化训练结果
    """
    # 创建一个包含3个子图的画布
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    # 左图：数据点和决策边界
    with torch.no_grad():
        # 生成一条连续的曲线用于显示分类边界
        x_line = torch.linspace(x_train.min()-1, x_train.max()+1, 200).reshape(-1, 1)
        y_line = model(x_line)
    
    # 分别绘制两类数据点
    class0_mask = (y_train == 0).squeeze()
    class1_mask = (y_train == 1).squeeze()
    
    ax1.scatter(x_train[class0_mask].numpy(), y_train[class0_mask].numpy(), 
                alpha=0.6, c='blue', label='类别0 (不及格)', s=30)
    ax1.scatter(x_train[class1_mask].numpy(), y_train[class1_mask].numpy(), 
                alpha=0.6, c='red', label='类别1 (及格)', s=30)
    
    # 绘制sigmoid曲线（决策边界）
    ax1.plot(x_line.numpy(), y_line.numpy(), 'g-', linewidth=2, label='决策曲线')
    ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='决策阈值 (0.5)')
    
    ax1.set_xlabel('学习时长 (小时)', fontsize=12)
    ax1.set_ylabel('预测概率', fontsize=12)
    ax1.set_title('逻辑回归分类结果', fontsize=14)
    ax1.legend(prop={'size': 10})
    ax1.grid(True, alpha=0.3)
    
    # 中图：训练过程中的损失值变化
    ax2.plot(losses, 'b-', linewidth=1)
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('损失值 (Loss)', fontsize=12)
    ax2.set_title('训练过程 - 损失值变化', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 右图：训练过程中的准确率变化
    ax3.plot(accuracies, 'r-', linewidth=1)
    ax3.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax3.set_ylabel('准确率 (%)', fontsize=12)
    ax3.set_title('训练过程 - 准确率变化', fontsize=14)
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_result.png', dpi=100, bbox_inches='tight')
    print(f"\n可视化结果已保存到: logistic_regression_result.png")
    plt.show()


def test_model(model):
    """
    测试模型：用训练好的模型进行预测
    """
    print(f"\n{'='*50}")
    print(f"模型测试 - 预测学生是否及格")
    print(f"{'='*50}")
    
    # 测试几个学习时长
    test_inputs = [2.0, 4.0, 5.0, 6.0, 8.0]
    
    with torch.no_grad():
        for hours in test_inputs:
            x_test = torch.tensor([[hours]])
            prob = model(x_test).item()  # 预测概率
            prediction = 1 if prob > 0.5 else 0  # 判断类别
            result = "及格 ✓" if prediction == 1 else "不及格 ✗"
            
            print(f"学习时长 {hours:.1f} 小时  →  预测概率: {prob:.4f}  →  预测结果: {result}")


def explain_sigmoid():
    """
    额外说明：Sigmoid函数的作用
    """
    print(f"\n{'='*50}")
    print(f"Sigmoid 函数说明")
    print(f"{'='*50}")
    print(f"公式: y = 1 / (1 + e^(-z))")
    print(f"\n具体例子：")
    
    # 几个示例值
    z_values = [-5, -2, 0, 2, 5]
    for z in z_values:
        sigmoid_value = 1 / (1 + np.exp(-z))
        print(f"  当 z = {z:2d} 时, sigmoid(z) = {sigmoid_value:.4f}")
    
    print(f"\n特点：")
    print(f"  - z很大时（如z=5），sigmoid接近1 → 判断为类别1")
    print(f"  - z很小时（如z=-5），sigmoid接近0 → 判断为类别0")
    print(f"  - z=0时，sigmoid=0.5 → 刚好在分界线上")


# 主程序
if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch 逻辑回归模型示例")
    print("=" * 50)
    
    # 训练模型
    model = train_model()
    
    # 测试模型
    test_model(model)
    
    # 解释Sigmoid函数
    explain_sigmoid()
    
    print(f"\n{'='*50}")
    print("程序执行完成！")
    print("=" * 50)
