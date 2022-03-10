# coding = utf-8
# @Time : 2022/1/12 18:30
# @Author : hky
# @File : code.py
# @Software : PyCharm

import numpy as np
import struct
import matplotlib.pyplot as plt
from template_attack.pca import *
from template_attack.file_operate import *
from template_attack.utils import *
import cpa.cpa_aes as AES
from cpa.cpa_SCAUtil import SBOX, HW, correlation


trace_length = 435002
raw_trace_num = 10000
start, end = 0, 1000


class TA:
    '''
    An implementation of TA on 1 Byte of AES, the leak model is Hamming Weight by default.
    '''
    leak_model = None
    leak_range = None
    pois = None
    mean_matrix = None
    cov_matrix = None
    mask = None
    mask_offset = None

    def __init__(self, traces, plain_texts, real_key, num_pois, mask, mask_offset, leak_model=HW, poi_spacing=20):
        [trace_num, trace_point] = traces.shape
        # print(traces.shape)
        self.leak_range = max(leak_model) + 1
        self.leak_model = leak_model
        self.mean_matrix = np.zeros((self.leak_range, num_pois))
        self.hw_cov_matrix = np.zeros((self.leak_range, num_pois, num_pois))
        self.mask = mask
        self.mask_offset = mask_offset
        # print(self.cov_matrix.shape)
        # temp_SBOX = [SBOX[plain_texts[i] ^ real_key] ^ mask[(offset + 1) % 16] for i in range(trace_num)]
        #这边，假设取了第一个plaintext的第一个字节，他得和第一个offset异或。p是第n个plaintext的第一个字节，o是第n个offset的偏移
        temp_SBOX = [SBOX[p ^ real_key] ^ self.mask[(o + 1) % 16] for p, o in zip(plain_texts, self.mask_offset)]
        # print(np.shape(temp_SBOX)) # 9800
        temp_lm = [leak_model[s] for s in temp_SBOX] # 计算理论上的泄露模型

        # Sort traces by HW
        # Make self.leak_range blank lists - one for each Hamming weight
        temp_traces_lm = [[] for _ in range(self.leak_range)]
        # print(temp_traces_lm) # 9个[]
        # Fill them up
        for i, trace in enumerate(traces):
            # print(i)
            # print(trace)#7个数
            temp_traces_lm[temp_lm[i]].append(trace) #把每个理论泄露值的曲线加到这个对应的hw的索引下
        # print(np.shape(temp_traces_lm))
        for mid in range(self.leak_range):
            assert len(temp_traces_lm[
                           mid]) != 0, "No trace with leak model value = %d, try increasing the number of traces" % mid

        # Switch to numpy arrays
        temp_traces_lm = [np.array(temp_traces_lm[_]) for _ in range(self.leak_range)]
        # print(temp_traces_lm[0].shape)# hw为0的里面有1条 1*84
        # Find averages
        tempMeans = np.zeros((self.leak_range, trace_point))
        for mid in range(self.leak_range):
            tempMeans[mid] = np.average(temp_traces_lm[mid], 0)#对每个hw的曲线进行平均值 比如hw为0的曲线，共5条，那就对这45条trace压缩成一条曲线的平均值，即只有一行了，84个点
        # print(np.shape(tempMeans))#(9, 84)

        # Find sum of differences
        tempSumDiff = np.zeros(trace_point)
        for i in range(self.leak_range):
            for j in range(i):
                # print(np.abs(tempMeans[i] - tempMeans[j]))
                tempSumDiff += np.abs(tempMeans[i] - tempMeans[j]) # 对任意两个平均值曲线求差
                # print(tempSumDiff)
        # print(tempSumDiff)

        # Find POIs
        self.pois = []
        for i in range(num_pois): # 5
            # Find the max
            nextPOI = tempSumDiff.argmax()
            self.pois.append(nextPOI) # 加入5个最大值
            # Make sure we don't pick a nearby value

            poiMin = max(0, nextPOI - poi_spacing)
            poiMax = min(nextPOI + poi_spacing, len(tempSumDiff))
            for j in range(poiMin, poiMax):
                tempSumDiff[j] = 0
        # print(self.pois) #[1, 13, 6, 19, 24]
        # Fill up mean and covariance matrix for each HW
        self.mean_matrix = np.zeros((self.leak_range, num_pois))
        self.cov_matrix = np.zeros((self.leak_range, num_pois, num_pois))
        for mid in range(self.leak_range):
            # print(mid)
            if temp_traces_lm[mid].shape[0] == 0:
                self.mean_matrix[mid] = np.zeros(num_pois)
                self.cov_matrix[mid] = np.zeros((num_pois, num_pois))
            elif temp_traces_lm[mid].shape[0] == 1: #仅一条曲线，那么就是1*84个点 需要求 i=1,j=1时，两个数的方差？共求25个这样的数
                self.mean_matrix[mid] = temp_traces_lm[mid][0][self.pois]
                self.cov_matrix[mid] = np.zeros((num_pois, num_pois))
            else:
                for i in range(num_pois):
                    # Fill in mean
                    self.mean_matrix[mid][i] = tempMeans[mid][self.pois[i]] #取hw为0的点 且坐标是1的点，也就是压缩后的平均值
                    for j in range(num_pois):
                        x = temp_traces_lm[mid][:, self.pois[i]] #mid为0时，是5*84的5条曲线，分别取里面的比如第1列和第1列，第一列和第13列等求方差。也就是5*1和5*1求。如果出现只有一条曲线或没有曲线咋办呢？？？
                        y = temp_traces_lm[mid][:, self.pois[j]]
                        # print(np.shape(x))
                        self.cov_matrix[mid, i, j] = cov(x, y) # 选取泄露模型里同一hw的值，比如0，然后取他们所有行中的某一列，也就是都是45*1的列向量
        print("The template has been created.")
        return

    def attack(self, traces, plaintext, init_index, cnt):
        # print(self.pois) #[1, 6, 0, 0, 0]
        rank_key = np.zeros(256)  # 评分表
        for j, trace in enumerate(traces): # 把每个trace挑出来
            # Grab key points and put them in a small matrix
            a = [trace[poi] for poi in self.pois]
            # print(a)
            # Test each key
            for k in range(256):
                # Find leak model coming out of sbox
                # 第一次循环 0, 1000个数
                # 第0个明文 以及 假设这个明文在原来的排列中是第351个，那么它后面这个掩码就得去offset[351]
                mid = self.leak_model[SBOX[plaintext[j] ^ k] ^ self.mask[(self.mask_offset[init_index + j] + 1) % 16]]

                # Find p_{k,j}
                # print(np.linalg.det(self.cov_matrix[mid]))
                rv = multivariate_normal(self.mean_matrix[mid], self.cov_matrix[mid], allow_singular=True) # 生成一个服从多元正态分布的数组
                p_kj = PRE[mid] * rv.pdf(a) # 概率密度函数
                # print(p_kj)

                # Add it to running total
                rank_key[k] += np.log(p_kj)



        guessed = rank_key.argsort()[-1]
        print("Key found: %d at %d round." % (guessed, cnt))
        return self.mean_matrix, self.cov_matrix, guessed


# 因为运行起来时间较长,所以已经运行过并且存储好了,这边先注释掉
print("**********处理前1000个曲线并存储至00000.npy文件*************")
print("**********已提前运行过,此处不展示具体过程**********************")
# raw_trace = np.array(read_trace_file(1000))
# print(raw_trace[:,:10])
# np.save("traces/00000.npy", raw_trace)
print("\n\n")


# 读取indexfile
def get_index_line(single_line):
    columns = single_line.split(" ")
    return columns[:-1] # except for the last column

def read_index_file(start :int, offset : int):
    index_lines = []
    with open("data/dpav4_rsm_index", "rb") as f:
        count = 0
        print("[*] Reading DPA index file from %d to %d" % (start, start + offset - 1))
        for line in f:
            if count < start:
                continue
            if count >= start + offset:
                break
            index_lines.append(get_index_line(line.decode()))
            count += 1
    assert len(index_lines) == offset
    return index_lines

def save_index_file(index_contents):
    crypto_ctnt, offset_ctnt = [], []
    showed_key = {}
    for content in index_contents:
        key_index = int(content[-1][1:])  # 00 01..., etc
        if key_index not in showed_key:
            showed_key[key_index] = content[0]
        crypto_ctnt.append([content[1], content[2]])
        offset_ctnt.append(content[3])

    np.save("data/cipher.npy", crypto_ctnt)
    np.save("data/offset.npy", offset_ctnt)
    np.save("data/key.npy", showed_key)
    return 1


def load_index_npy_file():
    crypto = np.load("data/cipher.npy")
    offset = np.load("data/offset.npy")

    # modify the default parameters of np.load
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)  # compromise
    showed_key = np.load("data/key.npy").item()
    np.load = np_load_old
    return crypto, offset, showed_key


def load_mask_file():
    mask = np.load("data/mask.npy")  # dpa contest v4.1 mask
    print("[*] The AES mask is :", mask)
    return mask


def process_index_file(start: int, offset: int):
    index_contents = read_index_file(start, offset)
    try:
        if save_index_file(index_contents):
            print("[*] Saving DPA index file done.")
    except Exception as e:
        print("[*] Exception:", e)
        return -1
    crypto, offset, showed_key = load_index_npy_file()  # load index data
    # dump some data to verify

    print("\t[*] plaintxt/ciphertxt samples: ", crypto[:1])
    print("\t[*] offset samples: ", offset[:5])
    print("\t[*] AES key:", showed_key[0])

    return 0


process_index_file(start, end)

# 获取基本信息
crypto, offset, showed_key = load_index_npy_file()
print("crypto的形状是", crypto.shape)
print("offset的形状是", offset.shape)
print("\n\n")

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
print("mask_offset的形状是", np.shape(mask_offset))
plaintext = []
for p in crypto[:, 0]:
    plaintext.append([int(p[b : b + 2], 16) for b in range(0, len(p), 2)])# 每两个字符组成一个十进制数
plaintext = np.array(plaintext)
print("plaintext的形状", plaintext.shape)
print("\n\n")

# 从npy文件读取曲线
def load_npy_traces(start, end):
    raw_samples = np.load("traces/00000.npy")
    raw_traces = np.array(raw_samples[start: start + end])
    print("Reading traces from %d to %d done." % (start, start + end))
    return raw_traces

raw_traces = load_npy_traces(start, end)
trace_num = raw_traces.shape[0] # 就等于end
sample_num = raw_traces.shape[1] # 435002
print("raw_traces的形状是", raw_traces.shape)
print("\n\n")




#以下是所有字节，尝试破解 228403

trace_num = 1000
train_key_array = np.array([108, 236, 198, 127, 40, 125, 8, 61, 235, 135, 102, 240, 115, 139, 54, 207])

#先进行相关性分析，挑出有用的几列
print("************相关性计算中***************")
max_corr_col = [228403, 309397, 194426, 277187, 257160, 233936, 194710, 218934, 285550, 183086, 237420, 332797, 130138, 168098, 195429, 244283]
# for plain_ind in range(16):
#     key_ind = train_key_array[plain_ind]
#     masked_state_byte_leak1 = np.array(
#         [HW[SBOX[p[plain_ind] ^ key_ind] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])
#     s1_corr_rank = np.zeros(sample_num)
#     candidate_traces = raw_traces[:trace_num]
#     s1_corr_rank += correlation(masked_state_byte_leak1[:trace_num], candidate_traces)
#     s1_ind = s1_corr_rank.argsort()[-1] #最大值出现的ind
#     max_corr_val = s1_corr_rank[s1_ind]
#     # print(s1_ind) #看一下最大值所在的trace的时间点,k为108的时候应该要更新到228403
#     max_corr_col.append(s1_ind)
#
max_corr_col = np.array(max_corr_col)
print(max_corr_col)
print("\n\n")

print("*************攻击密钥*****************")

# for key_ind in range(16):

key_ind = 0
traces = raw_traces[:, max_corr_col[key_ind] - 1000: max_corr_col[key_ind] + 1000]
# normalization
traces = standardize(traces)
#PCA
pca = PCA(traces, explain_ratio=0.95)
traces = pca.proj(traces)
# print(traces.shape)
num_train = 980

#将1000列对应的plaintext的点挑出来
singleplaintext = plaintext[:, key_ind]
# print(singleplaintext.shape)

# 分成train和attack组
# Train set
# 一共500条
train_tr = traces[:num_train, :]
train_pt = singleplaintext[:num_train]
# Attack set
attack_tr = traces[num_train:, :]
attack_pt = singleplaintext[num_train:]

train_key = train_key_array[key_ind]
print(train_key)

# Get a TA attacker
ta = TA(traces=train_tr, plain_texts=train_pt, real_key=train_key, num_pois=5, mask=mask, mask_offset=mask_offset)
mean_matrix, cov_matrix, guessed = ta.attack(attack_tr, attack_pt, init_index=num_train, cnt=key_ind)
#[108, 236, 198, 127, 40, 125, 8, 61, 235, 135, 102, 240, 115, 139, 54, 207]




