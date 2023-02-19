
file_name='all_pic.txt'
with open('train.txt', 'w') as train_txt:
    with open('test.txt', 'w') as test_txt:
        # 读取txt文件，每行按制表符分割提取数据
        with open(file_name, 'r') as file_txt:
            count = len(file_txt.readlines())   #获取文件总的行数
            print(count)
            file_txt.seek(0)  #回到文件开头
            for i in range(count):
                line_datas = file_txt.readline()
                if(i%20==0):   #每10个数据1个测试集
                    test_txt.write(line_datas)
                else:
                    train_txt.write(line_datas)
