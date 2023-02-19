import os
paths=r'./data'
f=open('all_pic.txt','wt',encoding='utf-8')   #all_pic502.txt'为500张所有数据集备份，可按自己情况更改

filenames=os.listdir(paths)
for filename in filenames:
    if os.path.splitext(filename)[1]=='.png':
        imgname=filename.split('.')[0]
        imgpath=r'./data'
        out_path=imgpath+'/'+imgname+' '+imgname
        print(out_path)
        #f.writable(out_path+'\n')
        f.write(out_path+'\n')

f.close()


