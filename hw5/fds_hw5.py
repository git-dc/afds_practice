
# coding: utf-8

# In[33]:


# AFDS Homework 5 --- Daniel Chirita (dc3316) --- April 5, 2019

# Problem 3 


## packages: 
import string, random
from tqdm import tnrange
from tqdm import tqdm_notebook as tqdm
from collections import deque

## helper functions:
def wordStream (fileName, sample_size=-1):
    wordcount = 0
    with open (fileName, 'r') as infile:
        for line in infile:
            for w in line.strip().lower().split():
                z = 0
                for c in w.strip (string.punctuation):
                    z = (z<<8) | ord(c)
                yield z
                wordcount +=1
            sample_size -= 1
            if sample_size == 0: break
#         print('total word count:', wordcount) # 1095695
# print('distinct words in big.txt:', countDistinct(wordStream('big.txt'))) # 38369

def shingleStream (fileName, shingle_size=9, sample_size=-1):
    shinglecount = 0
    with open (fileName, 'r') as infile:
        shingle = deque()
        for i in range(shingle_size):
            shingle.append(infile.read(1))
        z = 0
        for c in ''.join([i for i in shingle]):
            z = (z<<8) | ord(c)
        shinglecount += 1
        sample_size -= 1
        yield z
        
        while True:
            newChar = infile.read(1)
            if newChar == '\n': continue
            if newChar == '': break
            shingle.append(newChar)
            shingle.popleft()
            shinglecount +=1
            z = 0
            for c in ''.join([i for i in shingle]):
                z = (z<<8) | ord(c)
            yield z
            
            sample_size -= 1
            if sample_size == 0: break
#         print('total shingle count:', shinglecount) # 6360200
# print('distinct 9-shingles in big.txt:', countDistinct(shingleStream('big.txt',9))) # 2806810
                
def countDistinct (stream):
    M = {}
    for x in stream: M[x] = 1
    return len(M.keys())
    # distinct words in big.txt: 38369
    # distinct 9-shingles in big.txt: 2806810
    
def median (vals):
    return sorted(vals)[int(len(vals)/2)]

def mean (vals):
    return round(sum(vals)/len(vals))

def harmonic_mean (vals):
    return round(len(vals)/sum([i**-1 for i in vals]))

def sd (true_val, estimate):
    return round((true_val - estimate)**2 / true_val)

def perc_error (true_val, estimate):
    return round(100*abs(true_val - estimate)/true_val,2)

def hyperLogLog (vals):
    return round(len(vals)**2 / sum([i**-1 for i in vals]))


# ### Task:
# Write a function FM estimates which takes as input the stream and a number r denoting the required number of estimates and returns an array of independent estimates $[z_0 , z_1 , · · · , z_{r-1} ]$. There are several ways of combining the estimates to obtain a final estimate. Note that each $z_i$ represents an estimate of $2^{z_i}$ by the FM algorithm. Try the following ways of combining the estimates and state which seems to be the best:
# 
# | Estimate                        | Expression of estimate in terms of $z_i$       |
# |---------------------------------|------------------------------------------------|
# | Mean of the estimates           |$(2^{z_0} + · · · + 2^{z_99} )/100$.            |
# | Median of the estimates         |$median\{2^{z_0}, · · · , 2^{z_{99}}\}$.        |
# | Harmonic mean of the estimates  |$100/(2^{-z_0} + · · · + 2^{-z_{99}})$.         |
# | HyperLogLog                     |$10000/(2^{-z_0} + · · · + 2^{-z_{99}})$        |

# In[6]:


## FM_estimates:

def FM_estimates(stream, r): 
    '''
    r is the number of estimates to be made
    '''
    p = 9576890767 # large prime
    m = 2**32
    
    # generate a_i's and b_i's for r random hash fns
    a = [random.randint(1,p) for i in range(r)]
    b = [random.randint(0,p) for i in range(r)]
    z = [0 for i in range(r)]
    
    for word in tqdm(stream,total=1095695):
        for i in range(r):
            # random hash of word
            h = bin(((a[i] * (word % p) + b[i]) % p) % m)
            if h != bin(0):
                z[i] = max(z[i],str(h)[::-1].index('1')) 
    return z
            


# In[65]:


stream = wordStream('big.txt')
estimates = [2**z for z in FM_estimates(stream,100)] # 148 sec
estimates = [i for i in estimates if i!=1]
print(estimates, len(estimates))


# In[66]:


print(f'mean:          {mean(estimates)}, error: {perc_error(38369,mean(estimates))}%')
print(f'median:        {median(estimates)}, error: {perc_error(38369,median(estimates))}%')
print(f'harmonic mean: {harmonic_mean(estimates)}, error: {perc_error(38369,harmonic_mean(estimates))}%')
print(f'hyperLogLog:   {hyperLogLog(estimates)}, error: {perc_error(38369,hyperLogLog(estimates))}%')


# ### Results
# 
# Median seems to perform best.
# 
# | Method        | Result  | % Error  |
# |---------------|---------|----------|
# | Mean          | 114770  | 199.12%  |
# | Median        | 32768   | 14.6%    |
# | Harmonic mean | 27522   | 28.27%   |
# | HyperLogLog   | 2752168 | 7072.89% |
# 
# True count of distinct words in big.txt: 38369

# In[81]:


# using stochastic averaging:

def stoch_FM_estimates(stream, r): 
    '''
    2^r is the number of groups to be made
    '''
    p = 9576890767 # large prime
    m = 2**33
    
    # generate a and b for a random hash fn
    a = random.randint(1,p)
    b = random.randint(0,p)
    z = [0 for i in range(2**r)]
    
    for shingle in tqdm(stream,total=6360201):
        # random hash of shingle
        h = bin(((a * (shingle % p) + b) % p) % m)
        hstr = str(h)[2:]
        group = hstr[:r]
#         print(h,group)
        val = hstr[r:]

        if bin(int(val,2)) != bin(0):
            z[int(group,2)] = max(z[int(group,2)],val[::-1].index('1')) 
    return z


# In[82]:


stream = shingleStream('big.txt')
# I kept seeing 0's for the first 32 estimates, checked the hash values and the first bit seems to invariably be 1. Effectively, an r value of 6 results in 32 estimates instead of 64. So I increased m to be 2**33 and r to 7.
estimates = list(stoch_FM_estimates(stream,7)) # 34 sec
estimates = [2**z for z in estimates]


# In[83]:


print(estimates, len(estimates))
'''
Out: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 131072, 16384, 1048576, 16384, 65536, 16384, 16384, 65536, 262144, 32768, 16384, 8192, 32768, 32768, 32768, 65536, 131072, 65536, 32768, 32768, 32768, 131072, 131072, 65536, 16384, 65536, 32768, 32768, 131072, 524288, 131072, 32768, 524288, 1048576, 16384, 65536, 65536, 16384, 65536, 16384, 32768, 131072, 32768, 65536, 32768, 32768, 16384, 16384, 32768, 4194304, 16384, 32768, 131072, 524288, 65536, 32768, 16384, 16384, 32768, 32768, 32768, 65536, 16384, 16384] 128
'''
estimates = [i for i in estimates if i!=1] # first bit is always 1, so only second half of estimates is > 1 
print(estimates, len(estimates))
'''
Out: [131072, 16384, 1048576, 16384, 65536, 16384, 16384, 65536, 262144, 32768, 16384, 8192, 32768, 32768, 32768, 65536, 131072, 65536, 32768, 32768, 32768, 131072, 131072, 65536, 16384, 65536, 32768, 32768, 131072, 524288, 131072, 32768, 524288, 1048576, 16384, 65536, 65536, 16384, 65536, 16384, 32768, 131072, 32768, 65536, 32768, 32768, 16384, 16384, 32768, 4194304, 16384, 32768, 131072, 524288, 65536, 32768, 16384, 16384, 32768, 32768, 32768, 65536, 16384, 16384] 64
'''


# In[84]:


print(f'mean:          {mean(estimates)}, error: {perc_error(2806810,mean(estimates))}%')
print(f'median:        {median(estimates)}, error: {perc_error(2806810,median(estimates))}%')
print(f'harmonic mean: {harmonic_mean(estimates)}, error: {perc_error(2806810,harmonic_mean(estimates))}%')
print(f'hyperLogLog:   {hyperLogLog(estimates)}, error: {perc_error(2806810,hyperLogLog(estimates))}%')


# ### Results
# 
# HyperLogLog outperforms other methods of estimate combination:
# 
# | Method        | Result     | % Error  |
# |---------------|------------|----------|
# | Mean          | 170112     | 93.94%   |
# | Median        | 32768      | 98.83%   |
# | Harmonic Mean | 32573      | 98.84%   |
# | hyperLogLog   | 2084683    | 25.73%   |
# 
# True count of distinct 9-shingles in big.txt is 2806810.
