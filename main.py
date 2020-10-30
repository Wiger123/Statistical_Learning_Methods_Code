import time

# 主函数
if __name__ == '__main__':
    # 获取当前时间, 作为起始时间
    startTime = time.time()

    # 获取当前时间, 作为结束时间
    endTime = time.time()

    # 显示用时时长
    print('Time Span:', endTime - startTime)
