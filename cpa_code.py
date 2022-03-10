# coding = utf-8
# @Time : 2022/1/1 12:16
# @Author : hky
# @File : test.py
# @Software : PyCharm

import numpy as np
import struct
import matplotlib.pyplot as plt
# import cpa_aes as AES
# from cpa_SCAUtil import SBOX, HW, correlation
import cpa.cpa_aes as AES
from cpa.cpa_SCAUtil import SBOX, HW, correlation

trace_length = 435002
raw_trace_num = 10000
start, end = 0, 500
# 暂时使用前500个曲线

# 读取indexfile
def get_index_line(single_line):
    columns = single_line.split(" ")
    # print(columns)
    return columns[:-1]


def read_index_file(start: int, end: int):
    index_lines = []
    with open("data/dpav4_rsm_index", "rb") as f:
        count = 0
        print("Reading DPA index file from %d to %d" % (start, start + end - 1))
        for line in f:
            if count < start:
                continue
            if count >= start + end:
                break
            index_lines.append(get_index_line(line.decode()))
            count += 1
    assert len(index_lines) == end
    return index_lines


def save_index_file(index_contents):
    crypto_ctnt, offset_ctnt = [], []
    showed_key = {}
    for content in index_contents:
        # print(content)
        key_index = int(content[-1][1:])
        # print(key_index)
        if key_index not in showed_key:
            showed_key[key_index] = content[0]
        crypto_ctnt.append([content[1], content[2]])
        offset_ctnt.append(content[3])
    np.save("data/cipher.npy", crypto_ctnt)
    np.save("data/offset.npy", offset_ctnt)
    np.save("data/key.npy", showed_key)
    return 1


#从npy文件读取index file
def load_index_npy_file():
    crypto = np.load("data/cipher.npy")
    offset = np.load("data/offset.npy")

    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    showed_key = np.load("data/key.npy").item()
    np.load = np_load_old
    return crypto, offset, showed_key


def load_mask_file():
    mask = np.load("data/mask.npy")
    # print("The AES mask is :", mask)
    return mask


# 整合上面的函数
def process_index_file(start: int, end: int):
    index_contents = read_index_file(start, end)
    try:
        if save_index_file(index_contents):
            print("Saving DPA index file done.")
    except Exception as e:
        print("Exception:", e)
        return -1
    crypto, offset, showed_key = load_index_npy_file()

    print("plaintxt/ciphertxt samples: ", crypto[:2])
    print("offset samples: ", offset[:5])
    print("AES key:", showed_key[0])
    return 0

# print("**********处理index_file基础信息*************")
# process_index_file(start, end)
# print("\n\n")

def read_single_file(trace_file):
    f = open(trace_file, "rb")
    content = f.read()
    # 找到WAVEDESC字�?�串开始统计一些有用信�?
    base_offset = content.find(b"WAVEDESC")
    # print(base_offset)
    first_data_array_len = struct.unpack("<I", content[base_offset + 60: base_offset + 60 + 4])[0]  # 1st data array len
    data_points_num = struct.unpack("<I", content[base_offset + 116: base_offset + 116 + 4])[
        0]  # overall data points num
    # print(first_data_array_len, data_points_num)

    header_len = struct.unpack("<I", content[base_offset + 36: base_offset + 36 + 4])[0]
    data_offset = base_offset + header_len
    # 直接跳过�?357�?字节
    # print(data_offset, data_points_num)

    first_points_ind = struct.unpack("<I", content[base_offset + 124: base_offset + 124 + 4])[0]
    last_points_ind = struct.unpack("<I", content[base_offset + 128: base_offset + 128 + 4])[0]
    # print(first_points_ind, last_points_ind) #0 435001

    # read_data
    single_trace_content = []
    for i in range(data_points_num):
        point = struct.unpack("b", bytes([content[data_offset + first_points_ind + i]]))[0]
        single_trace_content.append(point)
    return single_trace_content


def read_trace_file(trace_num):
    raw_trace = []
    try:
        print("Reading trace file...")
        for i in range(trace_num):
            trace_file = "traces/00000/Z1Trace" + str(i).zfill(5) + ".trc"
            raw_trace.append(read_single_file(trace_file))
        print("Reading trace file done.")
        return raw_trace
    except Exception as e:
        print("Exception:", e)


print("**********处理trace曲线�?,�?trace1为例*************")
first_trace = np.array(read_single_file("traces/00000/Z1Trace00000.trc"))
print('The very first 10 power sample:', first_trace[:10])
# plt.plot(first_trace)
# plt.show()
print("\n\n")

# 因为运�?�起来时间较�?,所以已经运行过并且存储好了,这边先注释掉
print("**********处理�?1000�?曲线并存储至00000.npy文件*************")
print("**********已提前运行过,此�?�不展示具体过程**********************")
# raw_trace = np.array(read_trace_file(1000))
# print(raw_trace[:,:10])
# np.save("traces/00000.npy", raw_trace)
print("\n\n")


# 准备AES算法相关数据
# 明文 偏移 第一轮轮密钥

# 获取基本信息
crypto, offset, showed_key = load_index_npy_file()

# 计算轮密钥
key_size = len(showed_key[0]) # 64 hex
real_key = bytes.fromhex(showed_key[0])
print("密钥为：")
print([int(b) for b in real_key])
key_size = len(real_key) # 32个数字 第一轮只用到前16个数字


aes_cipher = AES.AES()
rounds = 14
expandedKeySize = 16 * (rounds + 1)
expandedKey = aes_cipher.expandKey(real_key, key_size, expandedKeySize)
round_key = aes_cipher.createRoundKey(expandedKey, 0) # 第一轮轮密钥，即前16个数字
# print(round_key)
first_key_byte = round_key[0] # 0x6c
first_round_key = [round_key[i * 4 + j] for j in range(4) for i in range(4)]
print('The First Key byte: ', first_key_byte)
print('The First Round Key: ', first_round_key)
print("\n\n")

# 读取明文与掩码偏移
mask = load_mask_file()
mask_offset = [int(o, 16) for o in offset] # 将16进制转化为十进制
plaintext = []
for p in crypto[:, 0]:
    plaintext.append([int(p[b : b + 2], 16) for b in range(0, len(p), 2)])# 每两个字符组成一个十进制数
plaintext = np.array(plaintext)


# 从npy文件读取曲线
def load_npy_traces(start, end):
    raw_samples = np.load("traces/00000.npy")
    raw_traces = np.array(raw_samples[start: start + end])
    print("Reading traces from %d to %d done." % (start, start + end))
    return raw_traces

raw_traces = load_npy_traces(start, end)
trace_num = raw_traces.shape[0] # 就等于off
sample_num = raw_traces.shape[1] # 435002
print("\n\n")

#
# # 以下是密钥正确时的一些操作，在攻击时，循环key值代入这个过程。取最大的corr值的key即可
# print("*********计算理论泄露值即HW（先取所有明文的第一字节）*************")
# # 计算泄露模型
# masked_state_byte_leak = np.array([HW[SBOX[p[0] ^ first_key_byte] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])
# print("泄露矩阵的形状：", masked_state_byte_leak.shape)
# print("取泄露矩阵前20个数字看一看", masked_state_byte_leak[:20])
# print("\n\n")


print("*************计算16个字节的泄露值和trace的相关性**************")
sample_length = trace_length
analysis_trace_num = 500
most_corr_point = 5 # 记录最有可能的前5的值
result = ""

for plain_ind in range(16):
    max_corr_k = np.zeros(256)  # 记录256个k值时，最大的相关性，最后那个最大的相关性的k有最大可能成为第一字节的密钥
    max_corr_t = 0.0
    max_corr_value = 0.0
    for k in range(256):
        masked_state_byte_leak1 = np.array(
            [HW[SBOX[p[plain_ind] ^ k] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])
        # print("**************泄露模型*****************")
        # print('Sample leakage：', first_byte_leak[:10], masked_state_byte_leak[:10])#这边就是每个明文的第一字节得到的hw
        # print("泄露矩阵的形状：", masked_state_byte_leak1.shape)
        # print("取泄露矩阵前20个数字看一看", masked_state_byte_leak1[:20])

        # 相关性寻找，计算500条trace的每一列，与理论泄露值进行相关系数的计算，取出相关系数最大的那些点的索引
        s1_corr_rank = np.zeros(sample_length)
        candidate_traces = raw_traces[:analysis_trace_num]
        s1_corr_rank += correlation(masked_state_byte_leak1[:analysis_trace_num], candidate_traces)
        s1_ind = s1_corr_rank.argsort()[-most_corr_point:][::-1]
        max_corr_k[k] += s1_corr_rank[s1_ind[0]]
        # if(s1_corr_rank[s1_ind[0]] > max_corr_value):
        #     max_corr_t = s1_ind[0]
        #     max_corr_value = s1_corr_rank[s1_ind[0]]
        # plt.plot(s1_corr_rank)
        # plt.show()
        # print(k)
    # print(max_corr_t) #看一下最大值所在的trace的时间点,k为108的时候应该要更新到228403
    max_corr_ind = max_corr_k.argsort()[-most_corr_point:][::-1]  # 将最大的几个相关性的key的索引值也就是key本身排列出来
    result += hex(max_corr_ind[0])[2:].zfill(2) #有个特殊值08 需要填满0 否则少一位
    print("The most possible key is: ", result)