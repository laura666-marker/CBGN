import networkx as nx
import time
import numpy as np
import heapq
import network_process as ns
import TARank as AUC
import matplotlib.pyplot as plt
import math

'''
选择候选节点集
'''
'''
创新：1.初始化节点排序是整个图的节点排序 而不是社区内的节点排序 这使得在社区内部也考虑了网络全局信息
    2.在进行候选节点选择时，由于初始化BGN是反向地加入节点，首先加入的是网络中最不重要的节点，所以在遍历一部分最不重要的节点后，
    剩余节点满足候选节点的数量要求后，直接将剩余节点加入候选节点集，不用再计算其加入网络的最大连通分量的大小，减小其时间复杂度的同时提高准确率。
'''

def sortNode(G):
    '''
    按度从大到小排序
    '''
    matG = []
    nodeG = nx.nodes(G)
    # print(nodeG)
    matG.append(nodeG)  # 加入节点列表
    degreeG = list(dict(nx.degree(G)).values())
    # print(degreeG)
    matG.append(degreeG)  # 加入度列表
    # print(matG) #1127, 1128, 1129, 1130, 1131, 1132, 1133, 0)), dict_values([38, 30, 22, 10, 21, 26, 8,
    matG = np.array(matG)
    result = matG.T[np.lexsort(-matG)].T  # 按最后一行的顺序逆序  # 按度从大到小排序
    # x.T 将x 转置   a.T[np.lexsort(a)].T按最后一行顺序排序
    # np.lexsort((b,a)) # Sort by a, then by b
    li = result[0]  # 返回与度从大到小的节点序号
    return li


def greedHeap(G, vrank, c_size):  # vrank节点排序序列  #
    '''

    :param G:
    :param vrank:
    :param c_size: 想要选取的候选节点数量  则进行Heap_BGN的节点有 len(G)-c_size
    :return:
    '''
    heap = []
    len_G = len(G)
    S = []  # 集合S存放每次选择的节点
    # !!!!!!!! 不是顺序递增 就用字典来存储
    # cost = [1]*len_G  # 加入节点之后的R值（鲁棒性值）
    cost = {node: 1 for node in G.nodes()}  # 节点的R值（鲁棒性值）
    # is_update = np.zeros(len_G)  #已更新为1

    is_update = {node: 0 for node in G.nodes()}  # 记录是否被更新

    # print('社区内节点：', G.nodes())

    # Cluster = [set()] * len_G  # 用以记录每一个集团的成员节点
    # Cluster = {node: set() for node in G.nodes()}
    Cluster = {i: set() for i in range(len_G)}

    # NodeCluster = [0]*len_G #记录每一个节点所属集团的ID
    NodeCluster = {node: 0 for node in G.nodes()}  # 记录每一个节点所属集团的ID

    newClusterId = 1
    maxcluster = 1  # 最大连通分量的大小
    sumcluster = 0
    '''
    for i in range(len_G):
        heapq.heappush(heap,(cost[i],i,vrank[i]))#建堆
    '''
    i = 0
    for node in G.nodes():
        heapq.heappush(heap, (cost[node], i, vrank[i]))  # 根据初始化排序 去构建堆
        i += 1
    # print '建堆完成'
    print(heap)
    while len(heap) > c_size:
        v = heapq.heappop(heap)  # 从堆中删除元素，返回值是堆中最小的元素
        nodev = v[2]  # 弹出被选择的节点
        # print('被弹出节点：',nodev)
        if is_update[nodev] == 1:
            NodeCluster, Cluster, newClusterId, maxcluster = chooseV(nodev, G, NodeCluster, Cluster, newClusterId,
                                                                     maxcluster)
            sumcluster = sumcluster + maxcluster
            S.append(nodev)
            is_update = {node: 0 for node in G.nodes()}  # 选完一个节点后更新状态为未更新
        else:  # 如果没有更新则更新
            cost[nodev] = updateCost(nodev, G, NodeCluster, cost, Cluster, maxcluster)  # 更新节点的cost
            is_update[nodev] = 1  # 标记为已更新
            heapq.heappush(heap, (cost[nodev], v[1], nodev))  # 放入堆
    rnum = (sumcluster - len_G) / len_G / len_G
    # print (rnum)
    # print(S)
    S = set(G.nodes) - set(S)  # 求剩下的相对重要的节点
    return S


def updateCost(nodev, G, NodeCluster, cost, Cluster, maxcluster):  # 更新节点的cost       计算最大连通分量的大小
    ci = set()  # 记录nodev可能连接的集团的ID集合
    for vj in nx.all_neighbors(G, nodev):
        if NodeCluster[vj] != 0:  # vj已属于某个集团
            ci.add(NodeCluster[vj])
    sumc = 1
    for name in ci:
        sumc = sumc + len(Cluster[name])
    if sumc < maxcluster:
        sumc = maxcluster
    cost[nodev] = sumc
    return cost[nodev]


def chooseV(nodev, G, NodeCluster, Cluster, newClusterId, maxcluster):  # 选择该节点:创建新集团 or 合并集团    维持派系
    ci = set()  # 记录vi可能连接的集团的ID集合
    for vj in nx.all_neighbors(G, nodev):
        if NodeCluster[vj] != 0:  # vj已属于某个集团
            ci.add(NodeCluster[vj])
    if len(ci) == 0:  # 即新加入的节点为孤立节点
        NodeCluster[nodev] = newClusterId  # 分配给新的集团
        Cluster[newClusterId] = set()  # 新集团
        Cluster[newClusterId].add(nodev)  # 将vi加入
        newClusterId = newClusterId + 1
    else:  # 加入的节点不是孤立节点
        minci = min(ci)  # 记录Ci中编号最小的集团ID，记为minci
        NodeCluster[nodev] = minci  # 节点vi所属集团的ID minci
        Cluster[minci].add(nodev)  # 将节点vi加入到编号为minci的集团中
        # 将Ci中所有集团的成员节点都合并，均放入minci中
        for name in ci:
            if name != minci:
                for nod in Cluster[name]:
                    NodeCluster[nod] = minci  # 修改集团索引
                    Cluster[minci].add(nod)
        if len(Cluster[minci]) > maxcluster:
            maxcluster = len(Cluster[minci])  # 最大连通分量的大小
    return NodeCluster, Cluster, newClusterId, maxcluster


'''  选择一部分节点 一部分不重要的节点'''
def imp_BGN(G,rank_dg,rank_ks,seed_size): # seed_size选择的种子节点数量
    eplision=0.1  # 控制参数
    len_G = len(G)
    S = []  # 集合S存放每次选择的节点
    G_R=G.copy()
    # !!!!!!!! 不是顺序递增 就用字典来存储
    cost = {node: 1 for node in G.nodes()}  # 节点的R值（鲁棒性值）thtea
    deg_ks={node:eplision*(rank_dg[node]+rank_ks[node]) for node in G.nodes()}  # 辅助判断  保证辅助判断小于1
    # is_affected={node:0 for node in G.nodes()}  # 初始化所有节点未被影响

    # is_update = {node: 0 for node in G.nodes()}  # 记录是否被更新

    # print('社区内节点：', G.nodes())

    # Cluster = [set()] * len_G  # 用以记录每一个集团的成员节点  # 错误用法,因为用*的话，赋予统一空间，修改时会修改所有!!!!  应该用列表生成器
    # Cluster={node:set() for node in G.nodes()}

    Cluster = {i: set() for i in range(len_G)}

    NodeCluster = {node: 0 for node in G.nodes()}  # 记录每一个节点所属集团的ID

    newClusterId = 1
    maxcluster = 1  # 最大连通分量的大小
    while len(G_R)>seed_size:
        for node in G_R.nodes():  # 对于即将要加入图的节点计算其鲁棒性值，选其鲁棒性值最小的  # ！！！！！修改不是所有节点都进行更新操作 受影响的节点才进行更新
            cost[node] = updateCost(node, G, NodeCluster, cost, Cluster, maxcluster)+deg_ks[node]  # 更新节点的cost
        # thtea=sorted(cost.items(),key=lambda x:x[1],reverse=True) #从大到小排列
        # node_min=thtea[0][0]
        node_min=min(cost.items(),key=lambda x:x[1])[0] # 取节点
        NodeCluster, Cluster, newClusterId, maxcluster = chooseV(node_min, G, NodeCluster, Cluster, newClusterId,maxcluster)  # 即添加节点
        # 添加节点需要计算哪些节点受到影响
        S.append(node_min)
        G_R.remove_node(node_min)
        cost.pop(node_min)
    S = list(set(G.nodes) - set(S))  # 剩余的节点就是相对重要的节点   大量节省了时间
    return S

def imp_BGN_BFS(G,auc,eplision,seed_size): # auc 分数进行辅助判断  seed_size 选择的种子节点数量  eplision 区分参数 对于不同的网络有不同的值
    # eplision=0.01  # 控制参数  根据网络变化
    # eplision=0.0001   # for email
    len_G = len(G)
    S = []  # 集合S存放每次选择的节点
    G_R=G.copy()
    # !!!!!!!! 不是顺序递增 就用字典来存储
    cost = {node: 1 for node in G.nodes()}  # 节点的R值（鲁棒性值）thtea

    Cluster = {i: set() for i in range(len_G)}
    NodeCluster = {node: 0 for node in G.nodes()}  # 记录每一个节点所属集团的ID

    newClusterId = 1
    maxcluster = 1  # 最大连通分量的大小
    # sumcluster = 0
    while len(G_R)>seed_size:
        for node in G_R.nodes():  # 对于即将要加入图的节点计算其鲁棒性值，选其鲁棒性值最小的  # ！！！！！修改不是所有节点都进行更新操作 受影响的节点才进行更新
            cost[node] = updateCost(node, G, NodeCluster, cost, Cluster, maxcluster)*(1+eplision*auc[node])  # 更新节点的cost
        # thtea=sorted(cost.items(),key=lambda x:x[1],reverse=True) #从大到小排列
        # node_min=thtea[0][0]
        node_min=min(cost.items(),key=lambda x:x[1])[0] # 取节点
        NodeCluster, Cluster, newClusterId, maxcluster = chooseV(node_min, G, NodeCluster, Cluster, newClusterId,maxcluster)  # 即添加节点
        # 添加节点需要计算哪些节点受到影响
        S.append(node_min)
        G_R.remove_node(node_min)
        cost.pop(node_min)
    S = list(set(G.nodes) - set(S))  # 剩余的节点就是相对重要的节点   大量节省了时间
    return S


def calcNode_E1(G,v): # 邻居节点的度的熵
    result = 0
    degree1 = 0
    for u in list(G.neighbors(v)):  # 求节点v的邻居节点的度之和
        degree1 += G.degree(u)

    for u in list(G.neighbors(v)):
        result += G.degree(u) / degree1 * math.log(G.degree(u) / degree1)
    return -1 * result

def calcNode_E2(G): # 用度计算
    all_degree = nx.number_of_edges(G) * 2  # 网络中所有节点度之和 边2的数量*2
    # node's information pi
    node_information={}
    for node in G.nodes():
        node_information[node]=0
        for u in nx.neighbors(G,node):
            node_information[node]+=-(G.degree(u)/all_degree)*math.log(G.degree(u)/all_degree)
    return node_information

def get_seeds_by_HeapBGN(g, com_nodes,init_rank,seed_candidate):
    '''
    根据传入的参数选择备选节点加入备选节点库  在每个社区中选择
    :param g: 整体的社交网络
    :param com_nodes: 当前社区
    inin_rank：初始化排序
    :return:
    注意：测试初始化排序是社区内部的还是整个图的 是整个图的还是只是社区内部的？？？
    '''

    seed = []  # 返回选取的高影响力种子节点
    initrank_c=[] # 社区内部的初始化排序
    # 根据节点获取图的子图 社区
    community_graph = nx.subgraph(g, com_nodes)
    # nx.draw(community_graph, with_labels=True)
    # plt.show()

    node_c=community_graph.nodes()
    # 找到社区内部对应的初始化排序
    for i in range(len(init_rank)):
        if init_rank[i] in node_c:
            initrank_c.append(init_rank[i])

    # size_g = len(community_graph)  # 计算社区大小
    # ！！！！计算需要多少节点传入候选节点=======公式==== 0.5*size_g  相对于每个社区大小选取
    # initial-BGN  在每个社区中进行 然后截取部分节点
    # !!!!! 因为最先选择的节点是最不重要的节点 节点需要反转  所以只对部分节点进行HeapBGN操作 剩下的直接传入候选节点集 不用再对剩下的重要节点排序
    # seed_len = 0.6 * size_g
    # initrank = list(sortNode(community_graph))  # 初始化排序   社区内部
    # initrank.reverse()  # 度从小到大的编号

    print('初始化排序：', initrank_c)
    seed = greedHeap(community_graph, initrank_c, seed_candidate)  # 在每个社区中
    return list(seed)

def get_seeds_by_impBGN(g, com_nodes,rank1,rank2,seed_candidate):  # 在社区内部执行改进的BGN
    '''

    :param g:
    :param com_nodes: 社区内的节点集
    :param rank1:
    :param rank2:
    :param seed_candidate: 候选种子节点
    :return:
    '''
    # 根据节点获取图的子图 社区
    community_graph = nx.subgraph(g, com_nodes)
    # size_g = len(community_graph)  # 计算社区大小
    # ！！！！计算需要多少节点传入候选节点=======公式==== 0.5*size_g  相对于每个社区大小选取
    # initial-BGN  在每个社区中进行 然后截取部分节点
    # !!!!! 因为最先选择的节点是最不重要的节点 节点需要反转  所以只对部分节点进行HeapBGN操作 剩下的直接传入候选节点集 不用再对剩下的重要节点排序
    # seed_candicate = 0.6 * size_g
    seed=imp_BGN(community_graph,rank1,rank2,seed_candidate)
    return seed

'''  选择一部分节点 一部分不重要的节点'''

def get_seeds_by_impBGNBFS(g, com_nodes,auc,eplision,seed_candidate): # auc 分数进行辅助判断  eplision 区分参数
    # 根据节点获取图的子图 社区
    community_graph = nx.subgraph(g, com_nodes)
    # size_g = len(community_graph)  # 计算社区大小
    # ！！！！计算需要多少节点传入候选节点=======公式==== 0.5*size_g  相对于每个社区大小选取
    # initial-BGN  在每个社区中进行 然后截取部分节点
    # !!!!! 因为最先选择的节点是最不重要的节点 节点需要反转  所以只对部分节点进行HeapBGN操作 剩下的直接传入候选节点集 不用再对剩下的重要节点排序
    # seed_candicate = 0.6 * size_g
    seed = imp_BGN_BFS(community_graph,auc,eplision,seed_candidate)
    return seed

if __name__ == '__main__':
    # 输入图
    # G = nx.karate_club_graph()
    eplision=0.1
    G=nx.read_edgelist('data/karate_edges.txt',nodetype=int)
    com_lists = ns.get_community(G)  # 获得社区列表
    print(com_lists)
    print(com_lists[0])
    start_time = time.process_time()
    # 传入整个图的初始化排序
    init_rank_G=list(sortNode(G))
    init_rank_G.reverse()
    com_seed = get_seeds_by_HeapBGN(G, com_lists[0],init_rank_G,3)
    end_time = time.process_time()
    print('-----------------------')
    print('HeapBGN候选节点degree:',com_seed)  # 选出的候选节点集
    print('运行时间：', end_time - start_time)

    print('------------------------')

    node_ks = nx.core_number(G)
    node_ks_min = min(node_ks.items(), key=lambda x: x[1])[1]
    node_ks_max = max(node_ks.items(), key=lambda x: x[1])[1]
    # node_ks=sorted(node_ks.items(),key=lambda x:x[1],reverse=True)

    print('--------求节点的熵1-邻居节点------------')
    node_information1 = {node: calcNode_E1(G, node) for node in G.nodes()}
    node_inf_min = min(node_information1.items(), key=lambda x: x[1])[1]
    node_inf_max = max(node_information1.items(), key=lambda x: x[1])[1]
    print('节点熵：', node_information1)
    # 归一化

    # 对k-shell 值归一化  不用排序
    print('节点k-shell:',node_ks)
    for node in node_ks:  # 遍历字典键
        node_ks[node] = float(node_ks[node] - node_ks_min) / (node_ks_max - node_ks_min)
    print('归一化节点k-shell:',node_ks)

    print('--------对节点熵1进行归一化------------')
    for node in node_information1:  # 遍历字典键
        node_information1[node] = float(node_information1[node] - node_inf_min) / (node_inf_max - node_inf_min)
    print(node_information1)

    print('初始化排序：')

    init_rank={node: eplision*(node_information1[node]+node_ks[node]) for node in G.nodes()}
    init_rank=sorted(init_rank.items(),key=lambda x:x[1])   # 从小到大排序
    print(init_rank)
    # print(init_rank[0])
    initrank=[]
    for node,score in init_rank: # 初始化排序 从小到大的节点列表
        initrank.append(node)
    print('初始化排序结果',initrank)


    seed_imp= get_seeds_by_impBGN(G,com_lists[0],node_information1,node_ks,3)
    seed_imp_heap=get_seeds_by_HeapBGN(G,com_lists[0],initrank,3)
    print('改进熵度BGN候选节点：',seed_imp)
    print('改进熵度Heap-BGN候选节点：',seed_imp_heap)

    auc_score = AUC.process(G)
    print('AUC分数：', auc_score)
    auc_min = min(auc_score.items(), key=lambda x: x[1])[1]
    auc_max = max(auc_score.items(), key=lambda x: x[1])[1]
    # 进行归一化
    print('---------------归一化AUC分数-----------')
    for node in auc_score:  # 遍历字典键
        auc_score[node] = float(auc_score[node] - auc_min) / (auc_max - auc_min)
    print(auc_score)

    seed_impBGNAUC=get_seeds_by_impBGNBFS(G,com_lists[0],auc_score,0.01,3)
    print('改进BGNAUC候选节点：',seed_impBGNAUC)

