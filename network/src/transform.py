import numpy as np
import os

#from numpy file to yuv420
def import_yuv420(file_path, width, height, frm_num=1):

    fp = open(file_path,'rb')
    pixnum = width * height

    data_type = np.uint8
    y = np.zeros(pixnum*frm_num, dtype=data_type)
    u = np.zeros(pixnum*frm_num // 4, dtype=data_type)
    v = np.zeros(pixnum*frm_num // 4, dtype=data_type)
    for i in range(0, frm_num):
        y[i*pixnum : (i+1)*pixnum] = np.fromfile(fp, dtype=data_type, count=pixnum, sep='')
        u[i*pixnum//4 : (i+1)*pixnum//4] = np.fromfile(fp, dtype=data_type, count=pixnum//4, sep='')
        v[i*pixnum//4 : (i+1)*pixnum//4] = np.fromfile(fp, dtype=data_type, count=pixnum//4, sep='')
    fp.close()
    y = y.reshape((frm_num, height, width))
    u = u.reshape((frm_num, height//2, width//2))
    v = v.reshape((frm_num, height//2, width//2))

    return y, u, v  # return frm_num * H * W

def output_block_yuv(file_path, width, height, block_size, numfrm=1):
    y, u, v = import_yuv420(file_path, width, height, numfrm)

    block_num_in_width = width // block_size
    block_num_in_height = height // block_size
    #print(block_num_in_width, block_num_in_height)
    for id, comp in enumerate([y, u, v]):
        if id == 0:
            comp_block_size = block_size
        else:
            comp_block_size = block_size // 2
        pad_comp = np.zeros((comp.shape[0], comp.shape[1], comp.shape[2]), dtype=np.uint8)
        pad_comp[:,:, :] = comp
        numfrm = comp.shape[0]

        block_list = []
        for f_num in range(numfrm):
            for i in range(block_num_in_height):
                for j in range(block_num_in_width):
                    if i == 10 and j == 2:
                        pad_comp[f_num, i * comp_block_size:(i + 1) * comp_block_size, j * comp_block_size:(j + 1) * comp_block_size] = np.flipud(np.fliplr(pad_comp[f_num, i * comp_block_size:(i + 1) * comp_block_size, j * comp_block_size:(j + 1) * comp_block_size]))
                    if i== 8 and j==18:
                        pad_comp[f_num, i * comp_block_size:(i + 2) * comp_block_size, j * comp_block_size:(j + 2) * comp_block_size] = np.flipud(np.fliplr(pad_comp[f_num, i * comp_block_size:(i + 2) * comp_block_size, j * comp_block_size:(j + 2) * comp_block_size]))
                    if i== 9 and j==12:
                        pad_comp[f_num, i * comp_block_size:(i + 1) * comp_block_size, j * comp_block_size:(j + 1) * comp_block_size] = np.flipud(np.fliplr(pad_comp[f_num, i * comp_block_size:(i + 1) * comp_block_size, j * comp_block_size:(j + 1) * comp_block_size]))
        
        
        if id == 0:
            pad_comp.reshape((height * width))
            y_new = pad_comp
        elif id == 1:
            pad_comp.reshape((height * width // 4))
            u_new = pad_comp
        else:
            pad_comp.reshape((height * width // 4))
            v_new = pad_comp

    print('shape of block_y', y_new.shape)
    print('shape of block_u', u_new.shape)
    print('shape of block_v', v_new.shape)

    # 将 YUV420 数据写入二进制文件
    with open("output.yuv", "wb") as f:
        # 写入 Y 分量（全分辨率）
        y = y_new.astype(np.uint8).tobytes()
        f.write(y)

        # 写入 U 分量（半分辨率）
        u = u_new.astype(np.uint8).tobytes()
        f.write(u)

        # 写入 V 分量（半分辨率）
        v = v_new.astype(np.uint8).tobytes()
        f.write(v)

    return 

if __name__ == "__main__":
    # 创建模型
    output_block_yuv(file_path='/home/qiujp/VTM236/runenv/Video/DIV2K/0100_2040x1356.yuv',width=2040,height=1356,block_size=64,numfrm=1)