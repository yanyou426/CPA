#!/usr/bin/env python
# coding: utf-8

# In[354]:


import numpy as np
import struct
import matplotlib.pyplot as plt
import aes as AES
from SCAUtil import SBOX, HW, correlation


# In[337]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[416]:


trace_length = 435002 # data point num
raw_trace_num = 10000 # trace number
start, offset = 0, 3000 # specific range for the traces and data to be used


# # 0. 处理曲线等文件
# 
# - 对包含有明密文、密钥、偏移的index文件 以及 曲线文件 进行解析
# - index文件以及曲线文件解析后存储为npy文件方便后续使用
# 
# ## 0.0 解析index file

# In[301]:


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
    crypto_ctnt, offset_ctnt = [],[]
    showed_key = {}
    for content in index_contents:
        key_index = int(content[-1][1:]) # 00 01..., etc
        if key_index not in showed_key:
            showed_key[key_index] = content[0]
        crypto_ctnt.append([content[1], content[2]])
        offset_ctnt.append(content[3])
    
    np.save("data/cipher.npy", crypto_ctnt)
    np.save("data/offset.npy", offset_ctnt)
    np.save("data/key.npy", showed_key)
    return 1


# ## 0.1 从npy读取index file

# In[313]:



def load_index_npy_file():
    crypto = np.load("data/cipher.npy")
    offset = np.load("data/offset.npy")
    
    # modify the default parameters of np.load
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) # compromise
    showed_key = np.load("data/key.npy").item()
    np.load = np_load_old
    return crypto, offset, showed_key

def load_mask_file():
    mask = np.load("data/mask.npy") # dpa contest v4.1 mask
    print("[*] The AES mask is :", mask)
    return mask


# ## 0.2 Wrap Up

# In[339]:


def process_index_file(start : int, offset : int):
    index_contents = read_index_file(start, offset)
    try:
        if save_index_file(index_contents):
            print("[*] Saving DPA index file done.")
    except Exception as e:
        print("[*] Exception:", e)
        return -1;
    crypto, offset, showed_key = load_index_npy_file() # load index data
    # dump some data to verify
    
    print("\t[*] plaintxt/ciphertxt samples: ", crypto[:1])
    print("\t[*] offset samples: ", offset[:5])
    print("\t[*] AES key:", showed_key[0])

    return 0

process_index_file(start, offset)


# ## 0.3 解析能量迹曲线

# In[308]:


def read_single_file(trace_file):
    f = open(trace_file, "rb")
    content = f.read()
    base_offset = content.find(b"WAVEDESC")
    #print(base_offset)
    first_data_array_len = struct.unpack("<I", content[base_offset + 60 : base_offset + 60 + 4])[0] # 1st data array len
    data_points_num = struct.unpack("<I", content[base_offset + 116 : base_offset + 116 + 4])[0] # overall data points num
    #print(first_data_array_len, data_points_num)
    assert first_data_array_len == data_points_num
    
    header_len = struct.unpack("<I", content[base_offset + 36 : base_offset + 36 + 4])[0]
    data_offset = base_offset + header_len
    #print(data_offset, data_points_num)
    assert data_offset + data_points_num == len(content) # first 357 bytes and then 435002 bytes
    
    first_points_ind = struct.unpack("<I", content[base_offset + 124 : base_offset + 124 + 4])[0]
    last_points_ind = struct.unpack("<I", content[base_offset + 128 : base_offset + 128 + 4])[0]
    assert last_points_ind - first_points_ind + 1 == data_points_num
    #print(first_points_ind, last_points_ind)
    
    # read_data
    single_trace_content = []
    for i in range(data_points_num):
        point = struct.unpack("b", bytes([content[data_offset + first_points_ind + i]]))[0]
        single_trace_content.append(point)
    return single_trace_content


def read_trace_file(trace_num):
    raw_trace = []
    try:
        print("[*] Reading trace file...")
        for i in range(trace_num):
            trace_file = "traces/00000/Z1Trace" + str(i).zfill(5) + ".trc"
            raw_trace.append(read_single_file(trace_file))
        return np.array(raw_trace)
    except Exception as e:
        print("[*] Exception:", e)
        
# dump some traces and plot some samples 
first_trace = np.array(read_single_file("traces/00000/Z1Trace00000.trc"))
print('[*] The very first 10 power sample:', first_trace[:10])
plt.plot(first_trace)


# ## 0.4 准备AES算法相关数据
# - 读取明文、偏移、密钥
# - 获取第一轮轮密钥
# - 计算目标泄露

# In[404]:


crypto, offset, showed_key = load_index_npy_file()


# - （可选）计算AES轮密钥
#     - 由于SCA往往针对第一轮进行攻击，而第一轮的轮密钥与原密钥相同，因此许多时候无需专门计算其他轮密钥

# In[410]:


# round key calculating
key_size = len(showed_key[0]) # 128 hex bit
real_key = bytes.fromhex(showed_key[0])
print([int(b) for b in real_key])
assert len(real_key) == key_size // 2 # the whole real long key bytes
key_size = len(real_key)

aes_cipher = AES.AES()
rounds = 14 # aes 256 rounds
expandedKeySize = 16 * (rounds + 1)
expandedKey = aes_cipher.expandKey(real_key, key_size, expandedKeySize)
round_key = aes_cipher.createRoundKey(expandedKey, 0) # the real key that will be xored with plaintext

first_key_byte = round_key[0] # first key byte still 0x6c
first_round_key = [round_key[i * 4 + j] for j in range(4) for i in range(4)] # same as the first 16 bytes of the raw key
print('[*] The First Key byte: ', first_key_byte)
print('[*] The First Round Key: ', first_round_key)


# - 读取明文与掩码偏移，并处理成可直接操作的int格式

# In[346]:


# get mask value
mask = load_mask_file()
mask_offset = [int(o, 16) for o in offset]

plaintext = []
for p in crypto[:,0]:
    plaintext.append([int(p[b : b + 2], 16) for b in range(0, len(p), 2)])
plaintext = np.array(plaintext)


# ## 0.5 从npy文件读取曲线
# - raw trace npy file (from Yuhang Ji)
# - npy曲线文件包含 raw_trace_num * trace_length 的曲线

# In[342]:


# read some raw_traces

def load_npy_traces(start, off):
    raw_samples = np.load("traces/00000.npy")
    raw_traces = np.array(raw_samples[start : start + off])
    print("[*] Reading traces from %d to %d done." % (start, start + off))
    return raw_traces
    
raw_traces = load_npy_traces(start, offset)
trace_num = raw_traces.shape[0]


# # 1 寻找曲线能量泄露点
# 
# - 恢复策略是将二阶CPA转化为对一阶CPA的攻击
#     - 针对$SBOX[x_{0} \oplus k_{0}] \oplus x_{1}$ 进行相关性分析
#     - 具体操作是先将该点进行拆分表示，并获取两点之间的泄露能量差
# - 因此，需要明确该如何拆分，并且在曲线上找到对应的点

# ## 1.0 计算目标操作字节的理论泄露
# - $x_{1}\oplus m_{1}$
# - $SBOX[x_{0} \oplus k_{0}] \oplus m_{1}$
#     - 理论泄露模型使用汉明重量

# In[350]:


# get x1 xored m1 leakage in a vector
first_byte_leak = np.array([HW[p[1] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])

# get SBOX[x0 xor k0] xor m1 leakage in a vector
masked_state_byte_leak = np.array([HW[SBOX[p[0] ^ first_key_byte] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])


print('[*] leakage calculated.')
print('[*] Sample leakage', first_byte_leak[:10], masked_state_byte_leak[:10])


# ## 1.1 相关性寻找
# - 在曲线上根据相关性大小寻找可能的泄露位置
# - 核心思想：
#     - 取出X条曲线的每一列，将其与理论泄露值进行相关系数的计算
#     - 选出相关系数最大的那些点的索引以备后用

# In[358]:


sample_length = trace_length # Here we use the whole trace
analysis_trace_num = 1000
attack_trace_num = trace_num - analysis_trace_num
correlated_candidates = 20


# In[360]:


x1_corr_rank = np.zeros(sample_length) # x1 xor m1 leakage rank
s1_corr_rank = np.zeros(sample_length) # SBox[x0 xor k0] xor m1 leakage rank

candidate_traces = raw_traces[:analysis_trace_num]  
x1_corr_rank += correlation(first_byte_leak[:analysis_trace_num], candidate_traces)
s1_corr_rank += correlation(masked_state_byte_leak[:analysis_trace_num], candidate_traces)

# let's pick the top 20 most related points
x1_ind = x1_corr_rank.argsort()[-correlated_candidates : ][::-1] # x1 xored m1 pos
s1_ind = s1_corr_rank.argsort()[-correlated_candidates : ][::-1] # sbox(x0 xored k0) xored m1 pos

print("[*] Done picking the leaked samples.")
print("[*] The most relavant leaked point of x1 xored m1 is at %d, with correlation coefficient %lf " % (x1_ind[0], x1_corr_rank[x1_ind[0]]))
print("[*] The most relavant leaked point of SBOX[x0 xored k0] xored m1 is at %d, with correlation coefficient %lf " % (s1_ind[0], s1_corr_rank[s1_ind[0]]))


# - 对rank进行可视化表示，以确认相关性大小

# In[361]:


# take a look 
plt.plot(x1_corr_rank)
plt.plot(s1_corr_rank)


# ## 1.2 得到两点曲线泄露差值
# - 根据等式$x_{1}\oplus m_{1} \oplus SBOX[x_{0} \oplus k_{0}] \oplus m_{0}$ = $x_{1} \oplus SBOX[x_{0} \oplus k_{0}]$可知
#     - 需要构建$x_{1}\oplus m_{1}$与$SBOX[x_{0} \oplus k_{0}] \oplus m_{1}$泄漏点之间的联系
#     - 使用曲线点之间的能量差值进行联系
# - 通过得到的前20个可能泄漏点，一条曲线可以得到20个曲线能量差值，这些差值都与密钥和S盒有关

# In[373]:


power_diff = [] # attack_trace_num * 20 list
for trace in raw_traces[-attack_trace_num:]:
    trace_leak_point = [abs(trace[id1] - trace[id2]) for id1, id2 in zip(s1_ind, x1_ind)]
    power_diff.append(trace_leak_point)
power_diff = np.array(power_diff)

print("[*] Power difference between the two leaked points Done.")
print('[*] Samples: \n', power_diff[:3])
print(power_diff.shape)


# # 2 攻击1字节密钥
# - 此时，已获得 attack_trace_num * 20的矩阵，该矩阵存储了与目标泄露点相关性最强的点之间的能量差值
# - 应遍历0-255之间的每种取值，根据明文计算其理论能量泄露，并与差值矩阵进行相关性比较，以筛选最有可能的密钥

# In[389]:


# let's attack
key_rank = np.zeros((256, correlated_candidates))
for candidate_key in range(256):
    candidate_leak = np.array([HW[SBOX[p[0] ^ candidate_key] ^ p[1]] for p in plaintext[-attack_trace_num:]])
    assert candidate_leak.shape[0] == attack_trace_num
    key_speified_candidate = np.zeros(correlated_candidates)
    for leak_point in range(correlated_candidates):
        key_speified_candidate[leak_point] = correlation(candidate_leak, power_diff[:,leak_point])
    key_rank[candidate_key] = key_speified_candidate
possible_key, possible_loc = np.where(key_rank == np.max(key_rank))
print('[*] Retrived First Key Byte is: ', hex(possible_key[0])[2:])


# In[393]:


# the most relavant leakage points index
plt.plot(key_rank[:, possible_loc[0]])


# # 3. 攻击第一轮16字节密钥
# - 将上述攻击过程进行组合，可恢复出完整的第一轮轮密钥
#     - 第一轮轮密钥即原256位密钥的前128位
# 

# In[415]:


sample_length = trace_length # Here we use the whole trace
analysis_trace_num = 1000
attack_trace_num = trace_num - analysis_trace_num
correlated_candidates = 20

# wrap all up and attack
print("[*] The Original Key is: ", ''.join(list(map(lambda x : hex(x)[2:], first_round_key))))
retrieved_key = ""
for kpos in range(len(first_round_key)):
    # get x_k+1 xored m_k+1 leakage in a vector
    target_byte_leak = np.array([HW[p[(kpos + 1) % 16] ^ mask[(o + kpos + 1) % 16]] for p, o in zip(plaintext, mask_offset)])

    # get SBOX[x_k xor k_k] xor m_k+1 leakage in a vector
    masked_state_byte_leak = np.array([HW[SBOX[p[kpos] ^ first_round_key[kpos]] ^ mask[(o + kpos + 1) % 16]] for p, o in zip(plaintext, mask_offset)])

    
    xk_corr_rank = np.zeros(sample_length) # x1 xor m1 leakage rank
    sk_corr_rank = np.zeros(sample_length) # SBox[x0 xor k0] xor m1 leakage rank

    candidate_traces = raw_traces[:analysis_trace_num]  
    xk_corr_rank += correlation(target_byte_leak[:analysis_trace_num], candidate_traces)
    sk_corr_rank += correlation(masked_state_byte_leak[:analysis_trace_num], candidate_traces)

    # let's pick the top 20 most relavant points
    xk_ind = xk_corr_rank.argsort()[-correlated_candidates : ][::-1] # x1 xored m1 pos
    sk_ind = sk_corr_rank.argsort()[-correlated_candidates : ][::-1] # sbox(x0 xored k0) xored m1 pos
    
    
    power_diff = [] # attack_trace_num * 20 list
    for trace in raw_traces[-attack_trace_num:]:
        trace_leak_point = [abs(trace[id1] - trace[id2]) for id1, id2 in zip(sk_ind, xk_ind)]
        power_diff.append(trace_leak_point)
    power_diff = np.array(power_diff)
    assert power_diff.shape == (attack_trace_num, correlated_candidates)
    
    
    key_rank = np.zeros((256, correlated_candidates))
    for candidate_key in range(256):
        candidate_leak = np.array([HW[SBOX[p[kpos] ^ candidate_key] ^ p[(kpos + 1) % 16]] for p in plaintext[-attack_trace_num:]])
        assert candidate_leak.shape[0] == attack_trace_num
        key_speified_candidate = np.zeros(correlated_candidates)
        for leak_point in range(correlated_candidates):
            key_speified_candidate[leak_point] = correlation(candidate_leak, power_diff[:,leak_point])
        key_rank[candidate_key] = key_speified_candidate
    possible_key, possible_loc = np.where(key_rank == np.max(key_rank))
    retrieved_key += hex(possible_key[0])[2:]
    print("\t[*] Retrieving: ",retrieved_key)


# In[ ]:




