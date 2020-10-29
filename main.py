import time
import perceptron_dichotomy as pd

# 主函数
if __name__ == '__main__':
    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 获取训练数据
    trainData, trainLabel = pd.loadData("G:\\Statistical-Learning-Method_Code-master\\Mnist\\mnist_train.csv")

    # 获取测试数据
    testData, testLabel = pd.loadData("G:\\Statistical-Learning-Method_Code-master\\Mnist\\mnist_test.csv")

    # 训练获得权重和正确率
    w, b = pd.perceptron(trainData, trainLabel, 20)

    # 测试获得正确率
    accuracyRate = pd.test(testData, testLabel, w, b)

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示正确率
    print('Accuracy Rate:', accuracyRate)

    # 显示用时时长
    print('Time Span:', endTime - startTime)
