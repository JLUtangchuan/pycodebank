import shutil
import os


def get_all_pth(pth):
    pths = []
    for root, dirs, files in os.walk(pth): 
        for dir in dirs: 
            # print(os.path.join(root, dir))
            pths.append(os.path.join(root, dir))
        # for file in files: 
        #     print(os.path.join(root, file))
    return pths

def remove_file(pth, target_dir):
    """删除目标文件夹下所有指定的文件夹
    """
    li = get_all_pth(pth)
    for i in li:
        if os.path.isdir(i):
            # remove
            if target_dir in i:
                print('[-] ', i)


if __name__ == '__main__':
    remove_file(r"D:\data\models\shapenet", r'runX\checkpoint')

