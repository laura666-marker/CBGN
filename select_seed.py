import network_process as ns
import networkx as nx
import math
import pre_selects as pst
import imp_celf as celf


def select_seeds(g,coms,k):  # 选择种子
    '''

    :param g: the network
    :param coms: 社区列表
    :return: 总的候选节点
    '''
    eplision = 0.1
    fin_seed=[]
    # -------------------------候选种子------------------------------
    node_ks = nx.core_number(g)
    node_ks_min = min(node_ks.items(), key=lambda x: x[1])[1]
    node_ks_max = max(node_ks.items(), key=lambda x: x[1])[1]
    # node_ks=sorted(node_ks.items(),key=lambda x:x[1],reverse=True)

    print('--------求节点的熵1-邻居节点------------')
    node_information1 = {node: pst.calcNode_E1(g, node) for node in g.nodes()}
    node_inf_min = min(node_information1.items(), key=lambda x: x[1])[1]
    node_inf_max = max(node_information1.items(), key=lambda x: x[1])[1]
    print('节点熵：', node_information1)
    # 归一化

    # 对k-shell 值归一化  不用排序
    print('节点k-shell:', node_ks)
    for node in node_ks:  # 遍历字典键
        node_ks[node] = float(node_ks[node] - node_ks_min) / (node_ks_max - node_ks_min)
    print('归一化节点k-shell:', node_ks)

    print('--------对节点熵1进行归一化------------')
    for node in node_information1:  # 遍历字典键
        node_information1[node] = float(node_information1[node] - node_inf_min) / (node_inf_max - node_inf_min)
    print(node_information1)

    '''   初始化排序  '''
    # init_rank = {node: eplision * (node_information1[node] + node_ks[node]) for node in G.nodes()}
    # init_rank = sorted(init_rank.items(), key=lambda x: x[1])  # 从小到大排序  先选择不重要的节点
    # print(init_rank)
    # # print(init_rank[0])
    # initrank = []
    # for node, score in init_rank:  # 初始化排序 从小到大的节点列表
    #     initrank.append(node)

    g_num=len(g)
    threshold = g_num * 0.01
    Beta=1  # 放大因子   放大因子 调参 ！！！！！
    cand_seeds=[] # 记录总候选节点
    com_min=len(coms[0])
    com_max=com_min

    for com in com_lists:
        len_com=len(com)
        if len_com<com_min:
            com_min=len_com
        if len_com>com_max:
            com_max=len_com

    for com in coms:
        com_num=len(com)
        if com_num<threshold:  # 如果某个社区节点数小于某一阈值则跳过该社区 不在该社区中选择
            continue
        # 社区内选点！！！！！修改

        # sel_num = math.ceil(com_num / g_num * 2 * k)  # ceil() 方法执行的是向上取整计算，它返回的是大于或等于函数。  根据社区大小定初始种子
        sel_num=math.ceil((com_num-com_min)/(com_max-com_min)*Beta*k)
        cand_seed=pst.get_seeds_by_impBGN(g,com,node_information1,node_ks,sel_num)  # imp_BGN
        # cand_seed=pst.get_seeds_by_HeapBGN(g,com,initrank,sel_num)   # 堆加速
        cand_seeds.extend(cand_seed)
    print('候选种子选择完成!!!')
    print('候选节点为',cand_seeds)
    print('---------------选择种子节点----------------')
    fin_seed,avg_inf=celf.call_celf(g,k,cand_seeds,node_ks,20)
    return fin_seed


if __name__ == '__main__':
    G = nx.read_edgelist('data/Jazz', nodetype=int)
    com_lists = ns.get_community(G)  # 获得社区列表
    seeds=select_seeds(G,com_lists,6) # 选择5个种子节点的候选节点集   集合是无序的
    print('选择种子节点为:',seeds)

    # CENEW [2, 4, 5, 7, 214]