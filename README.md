# OCR项目实战（一）：手写汉语拼音识别

项目介绍：

本项目基于深度学习的手写汉语拼音识别方法研究与实现。项目采用Pytorch框架，整体采用主流深度学习文字识别算法CRNN+CTC方法，项目流程主要分为数据集采集及标注，算法构建、模型训练、预测与评估等。后续会补充PaddleOCR版本的手写汉语拼音识别，将引入更多模型测试，并结合数据增强手段提升模型泛化性。


📝项目讲解：https://blog.csdn.net/qq_36816848/article/details/128951065

✨欢迎订阅专栏，欢迎加群互相交流，q群：704932595 ,  群内将分享更多大数据与人工智能专业资料!!!!

![在这里插入图片描述](https://img-blog.csdnimg.cn/785fed7a6b4c4dc5a18e33bd71a37907.png)



1.首先将制作好的图片放入data目录下，图片名按具体写的拼音命名，格式jpg。
2.执行pic_to_txt.py文件，生成用于文字识别的图片及标注信息all_pic.txt，内容需要包含图片路径名+拼音，\t分割。
3.运行split.py数据集脚本将图片总数量按9:1比例 (将all_pic.txt分别生成train.txt 和test.txt)
4.将txt格式转为lmdb格式数据集执行create_lmdb，得到train和testd lmdb文件夹，将两个路径替换train.py里的训练及测试路径。
5.运行train.py训练，跑一定时间将模型保存运行demo进行测试。

算法介绍：

1.CRNN+CTC  (vgg为特征提取网络)

![image](https://user-images.githubusercontent.com/30800097/219945486-bd6c9ac5-ebe3-47ec-bd84-32a8bb107b2b.png)



2.CRNN+Attention (resnet为特征提取网络)

![image](https://user-images.githubusercontent.com/30800097/219945469-f4d7ab0c-2808-4629-96f3-9516dd71716e.png)

欢迎Fork，后续将更新更多CV相关项目！
