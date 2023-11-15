# import torch,regex,numpy as np
from fastai.vision.all import *
import regex
# import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm

# 需要可视化
# 需要根据train.csv调整measurements列表
measurements = {
    'weight': [('mg',1), ('g', 1000), ('gr', 1000), ('gram', 1000), ('kg', 1000000)],
    'length': [('mm',1), ('cm', 10), ('m',1000), ('meter', 1000)],
    'pieces': [ ('pc',1)],
    'memory': [('gb', 1)],
    'volume': [('ml', 1), ('l', 1000), ('liter',1000)]
}

def to_num(x, mult=1):
    x = x.replace(',','.')
    return int(float(x)*mult)

def extract_unit(tit, m):
    pat = f'\W(\d+(?:[\,\.]\d+)?) ?{m}s?\W'
    matches = regex.findall(pat, tit, overlapped=True)
    return set(matches)

def extract(tit):
    res =dict()
    tit = ' '+tit.lower()+' '
    for cat, units in measurements.items():
        cat_values=set()
        for unit_name, mult in units:
            values = extract_unit(tit, unit_name)
            values = {to_num(v, mult) for v in values}
            cat_values = cat_values.union(values)
        if cat_values:
            res[cat] = cat_values
    return res


def add_measurements(data):
    data['measurement'] = data.title.map(extract)
    return data

def match_measures(m1, m2):
    k1,k2 = set(m1.keys()), set(m2.keys())
    common = k1.intersection(k2)
    if not common: return True
    for key in common:
        s1,s2 = m1[key], m2[key]
        if s1.intersection(s2):
            return True
    return False

def check_measurements(combined_dists, combined_inds, data_df):
    K = min(8, len(data_df)) * len(data_df)
    _, inds_k = combined_dists.view(-1).topk(K)
    removed = 0
    inds_k = inds_k.tolist()
    for idx in inds_k:
        x = idx // combined_inds.shape[1]
        y_idx = idx % combined_inds.shape[1]
        y = combined_inds[x,y_idx] 
        if not match_measures(data_df.iloc[x].measurement, data_df.iloc[y.item()].measurement):
            removed +=1
            combined_dists[x][y_idx]=0
    print('removed', removed, 'matches')
    
    
    

def add_target_groups(data_df, source_column='label_group', target_column='target'):
    target_groups = data_df.groupby(source_column).indices
    data_df[target_column]=data_df[source_column].map(target_groups)
    return data_df

def get_targets_shape(train_df): #获取每一组的长度和
    all_targets = add_target_groups(train_df).target.to_list() #获取每一个训练样本的答案
    # all_targets=list[train_df.target.split()]
    all_targets_lens = [len(t) for t in all_targets]
    targets_shape = []
    for size in range(min(all_targets_lens), max(all_targets_lens)+1):
        count = all_targets_lens.count(size) / len(all_targets)
        targets_shape.append((size,count))
    return targets_shape

def chisel(groups, groups_p, pos, target_count):
    probs = []
    groups_lens = [len(g)for g in groups]
    current_count = groups_lens.count(pos)
    if current_count >= target_count:
        return
    # 该长度还不够，把长度大于该长度的组按可能性顺序截断到该长度，直至符合要求
    to_cut = target_count - current_count
    for i in range(len(groups)):
        if len(groups_p[i])>pos:
            probs.append((i, groups_p[i][pos])) # 看该组的第pos个样本的概率，越小就越被截断
    probs.sort(key=lambda x:x[1])
    for i in range(min(to_cut, len(probs))):
        group_idx = probs[i][0] 
        groups[group_idx]=groups[group_idx][:pos]
        groups_p[group_idx]=groups_p[group_idx][:pos]
        

def sorted_pairs(distances, indices):
    triplets = []
    n= len(distances)
    for x in range(n):
        used=set()
        for ind, dist in zip(indices[x].tolist(), distances[x].tolist()):
            if not ind in used:
                triplets.append((x, ind, dist))
                used.add(ind)
    return sorted(triplets, key=lambda x: -x[2])

def do_chunk(embs):
    step = 1000
    for chunk_start in range(0, embs.shape[0], step):
        chunk_end = min(chunk_start+step, len(embs))
        yield embs[chunk_start:chunk_end]
        
def get_nearest(embs, emb_chunks, K=None, sorted=True):
    if K is None:
        K = min(51, len(embs))
    distances = []
    indices = []
    for chunk in emb_chunks:
        sim = embs @ chunk.T
        top_vals, top_inds = sim.topk(K, dim=0, sorted=sorted)
        distances.append(top_vals.T)
        indices.append(top_inds.T)
    return torch.cat(distances), torch.cat(indices)

def combined_distances(embs_list):
    lenn=len(embs_list[0])
    K = min(lenn, 51)
    combined_inds =[get_nearest(embs, do_chunk(embs))[1] for embs in embs_list] 
    combined_inds = torch.cat(combined_inds, dim=1) # n*(2K),第i行返回第i个样本的最近邻索引前n个是img，后n个是bert
    res_inds,res_dists = [],[]
    for x in range(len(combined_inds)):
        inds = combined_inds[x].unique()
        # inds=torch.arange(lenn)
        Ds = [embs[None,x] @ embs[inds].T for embs in embs_list] # 将 None 插入其中，可以将 embs[x] 变为一个行向量（1维数组），而不是一个标量（0维数组），
        D = Ds[0] + Ds[1] - Ds[0] * Ds[1] #Ds[0][i]表示i到其他向量的image余弦相似度，将余弦相似度看作概率
        top_dists, top_inds = D.topk(K) #取最大K个值和索引
        res_inds.append(inds[top_inds])
        res_dists.append(top_dists)
    return torch.cat(res_inds), torch.cat(res_dists)

def blend_embs(embs_list, data_df, threshold=.97, m2_threshold=.6):
    combined_inds, combined_dists = combined_distances(embs_list)
    # check_measurements(combined_dists, combined_inds, data_df)
    new_embs_list = L((torch.empty_like(embs) for embs in embs_list))
    for x in range(len(embs_list[0])):
        neighs = combined_dists[x] > threshold 
        if neighs.sum() == 1 and combined_dists[x][1]>m2_threshold: # 只有neighs[0]=1
            neighs[1]=1
        neigh_inds, neigh_ratios = combined_inds[x, neighs], combined_dists[x,neighs]
        for embs, new_embs in zip(embs_list, new_embs_list):
            new_embs[x] = (embs[neigh_inds] * neigh_ratios.view(-1,1)).sum(dim=0)
    return new_embs_list.map(F.normalize)