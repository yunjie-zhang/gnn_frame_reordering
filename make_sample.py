import numpy as np
import random
import torch
import dgl
import sys
import os

cat_map = {
                "网服": 0,
                "游戏": 1,
                "电商零售": 2,
                "本地": 3,
                "金融": 4,
                "交通类": 5,
                "直营电商": 6,
                "食品饮料类": 7,
                "母婴护理类": 8,
                "教育类": 9,
                "运动娱乐休闲类": 10,
                "美妆日化类": 11,
                "金融服务类": 12,
                "生活服务类": 13,
                "家居建材类": 14,
                "面向企业类": 15,
                "旅游类": 16,
                "医疗类": 17,
                "服饰类": 18,
                "房产类": 19,
                "游戏类": 20,
                "IT消电类": 21,
                "电商零售类": 22,
                "资讯类": 23,
                "高等教育": 24,
                "生活用品": 25,
                "招聘": 26,
                "母婴服务": 27,
                "装修服务": 28,
                "其他面向企业类": 29,
                "出行服务(含物流)": 30,
                "鞋/箱包": 31,
                "交易平台": 32,
                "3C": 33,
                "棋牌捕鱼": 34,
                "工具": 35,
                "休闲益智": 36,
                "其他旅游类": 37,
                "职业技能培训": 38,
                "饰品/配饰": 39,
                "工农业": 40,
                "射击游戏": 41,
                "金融综合线上平台": 42,
                "综合电商": 43,
                "医疗机构": 44,
                "留学/语言培训": 45,
                "招商加盟": 46,
                "民办学校": 47,
                "保险": 48,
                "家具家装综合": 49,
                "影音娱乐": 50,
                "保姆家政": 51,
                "其他资讯类": 52,
                "服装": 53,
                "商业零售": 54,
                "信用卡": 55,
                "其他食品饮料类": 56,
                "其他房产类": 57,
                "综合游戏平台": 58,
                "教育类平台": 59,
                "运营商": 60,
                "车类配件及周边": 61,
                "机械设备": 62,
                "饮料/茶/酒水": 63,
                "家具灯饰": 64,
                "电商商家": 65,
                "儿童/学生培训": 66,
                "冲印摄影": 67,
                "箱包": 68,
                "体育器械": 69,
                "第三方支付": 70,
                "宠物服务": 71,
                "策略游戏": 72,
                "小额贷款": 73,
                "丽人美发": 74,
                "机动车销售与服务": 75,
                "药品/保健品": 76,
                "医疗美容": 77,
                "跨境": 78,
                "食品": 79,
                "房地产": 80,
                "汽车资讯与服务平台": 81,
                "生活/健康": 82,
                "配饰": 83,
                "传统旅行社": 84,
                "医疗平台": 85,
                "其他直营电商": 86,
                "旅游景点": 87,
                "其他生活服务类": 88,
                "其他家居建材类": 89,
                "养成游戏": 90,
                "其他电商零售类": 91,
                "商务服务": 92,
                "婚恋/交友": 93,
                "证券": 94,
                "在线旅游服务平台": 95,
                "其他服饰类": 96,
                "休闲装/正装": 97,
                "运势测算": 98,
                "家装建材": 99,
                "酒店": 100,
                "化妆品": 101,
                "新闻": 102,
                "角色扮演": 103,
                "家纺家饰": 104,
                "其他运动娱乐休闲类": 105,
                "婚庆交友": 106,
                "内衣": 107,
                "展览": 108,
                "护肤品/化妆品": 109,
                "团购/折扣": 110,
                "餐饮服务": 111,
                "食品零食": 112,
                "节能环保": 113,
                "其他美妆日化类": 114,
                "银行服务": 115,
                "其他教育类": 116,
                "家用电器": 117,
                "洗涤/卫生用品": 118
}


FRAME_CNT = 8

def makePicPair(total_cnt, interval, ratio):
    base_pair_list = []
    pair_list = []
    label_list = []

    for i in range(interval):
        for j in range(interval):
            if i != j:
                base_pair_list.append([i, j])
    for i in range(total_cnt):
        base_idx = i * interval
        for base_pair in base_pair_list:
            pair_list.append([base_pair[0] + base_idx, base_pair[1] + base_idx])
            if base_pair[0] <= base_pair[1]:
                label_list.append(1.0)
            else:
                label_list.append(0.0)

    pair_cnt = len(pair_list)
    print("A total of {} pairs.".format(pair_cnt))
    ret_torch = torch.tensor(pair_list, dtype=torch.int)
    ret_torch = torch.transpose(ret_torch, 0, 1)

    label_torch = torch.tensor(label_list, dtype=torch.float)
    #label_torch = torch.transpose(label_torch, 0, 1)

    train_mask_torch = torch.ones(pair_cnt, dtype=torch.bool)
    test_idx = int(pair_cnt * ratio)
    for i in range(test_idx, pair_cnt):
        train_mask_torch[i] = False


    print("Size of torch tensor is {} {} {}".format(ret_torch.shape, label_torch.shape, train_mask_torch.shape))
    for i in range(20):
        print(int(ret_torch[0][i]), end="\t")
    print("\n")
    for i in range(20):
        print(int(ret_torch[1][i]), end="\t")
    print("\n")
    for i in range(20):
        print(int(ret_torch[0][i]), int(ret_torch[1][i]), float(label_torch[i]), int(train_mask_torch[i]), end="\t")
    print("\n")
    return ret_torch, label_torch, train_mask_torch

def loadTsv(tsv_path: str, video_name_list_full):
    #video_id -> account
    video_name_list_set = set(video_name_list_full)
    video_from_tsv = set()

    account2video = dict()
    account2info = dict()
    with open(tsv_path, "r") as fp:
        line = fp.readline()#the first line is title
        line = fp.readline()
        while line:
            fields = line.split("\t")
            if len(fields) < 2:
                line = fp.readline()
                continue

            account_str = fields[0]
            video_id_str = fields[1]
            cat_0 = fields[2]
            cat_1 = fields[3]
            cat_2 = fields[4]
            info_str = "_".join([cat_0, cat_1, cat_2])
            if video_id_str not in video_name_list_set:
                line = fp.readline()
                continue

            if account_str not in account2video:
                account2video[account_str] = []
            account2video[account_str].append(video_id_str)
            account2info[account_str] = info_str
            video_from_tsv.add(video_id_str)
            line = fp.readline()
#    for key in ret_dict.keys():
#        print("{}\t{}".format(key, len(ret_dict[key])))

    video_name_list = []
    video_candidate_set = video_from_tsv & video_name_list_set

    for video_id in video_candidate_set:
        video_name_list.append(video_id)


    return account2video, account2info, video_name_list

def makeAccPair(account2video, video_name2idx):
    account2idx = dict()
    idx2account = dict()
    acc_idx = 0
    for acc_str in account2video.keys():
        account2idx[acc_str] = acc_idx
        idx2account[acc_idx] = acc_str
        acc_idx += 1

    account2video_list = []
    for acc in account2video.keys():
        for video_id in account2video[acc]:
            account2video_list.append([acc, video_id])

    account2pic_list = []
    for acc_video_pair in account2video_list:
        acc_id = acc_video_pair[0]
        video_id = acc_video_pair[1]
        for pic_id in video_name2idx[video_id]:
            account2pic_list.append([acc_id, pic_id])

    acc_idx2pic_list = []
    pic2acc_idx_list = []
    for pair in account2pic_list:
        acc_idx = account2idx[pair[0]]
        pic_idx = pair[1]
        acc_idx2pic_list.append([acc_idx, pic_idx])
        pic2acc_idx_list.append([pic_idx, acc_idx])

    ret_torch_1 = torch.tensor(acc_idx2pic_list, dtype=torch.int)
    ret_torch_1 = torch.transpose(ret_torch_1, 0, 1)
    print("Size of torch tensor is {}".format(ret_torch_1.shape))

    ret_torch_2 = torch.tensor(pic2acc_idx_list, dtype=torch.int)
    ret_torch_2 = torch.transpose(ret_torch_2, 0, 1)
    print("Size of torch tensor is {}".format(ret_torch_2.shape))

    return ret_torch_1, ret_torch_2, account2idx, idx2account






def make_graph(tsv_path:str, feature_path: str):
    file_list = os.listdir(feature_path)
    video_name_list_full = [cur_str.split(".")[0] for cur_str in file_list]





    account2video, account2info, video_name_list = loadTsv(tsv_path, video_name_list_full)

    print("A total of {} accounts found.".format(len(account2video)))
    print("A total of {} accounts found.".format(len(account2info)))
    print("A total of {} videos found.".format(len(video_name_list)))
    idx2video_name = dict()
    video_name2idx = dict()

    video_idx = 0
    video_cnt = 0
    for i in range(len(video_name_list)):
        cur_video_id = video_name_list[i]
        for j in range(FRAME_CNT):
            idx2video_name[video_idx + j] = cur_video_id
        video_cnt += 1
        video_idx += FRAME_CNT

    for key in idx2video_name.keys():
        cur_video_id = idx2video_name[key]
        if cur_video_id not in video_name2idx:
            video_name2idx[cur_video_id] = []
        video_name2idx[cur_video_id].append(key)#key is the index here

    for cur_video_id in video_name2idx:
        if len(video_name2idx[cur_video_id]) != FRAME_CNT:
            print("Frame count doesn't match.")



    ret_acc2pic_node, ret_pic2acc_node, account2idx, idx2account = makeAccPair(account2video, video_name2idx)
    ret_pic_node, label_torch, train_mask_torch = makePicPair(video_cnt, FRAME_CNT, 0.9)#who connects whom, who leads whom, training label

    g = dgl.heterograph({('pic', 'nb', 'pic'): (ret_pic_node[0], ret_pic_node[1]), #neighbor
                         ('acc', 'pb', 'pic'): (ret_acc2pic_node[0], ret_acc2pic_node[1]), #publish
                         ('pic', 'blt', 'acc'): (ret_pic2acc_node[0], ret_pic2acc_node[1])}) #belongs to
    #g = dgl.heterograph({('pic', 'nb', 'pic'):(torch.tensor([0, 0, 2]), torch.tensor([1, 2, 3])), ('acc', 'own', 'pic'):(torch.tensor([0, 1, 0]), torch.tensor([1, 2, 3]))})
    pic_node_num = g.num_nodes('pic')
    acc_node_num = g.num_nodes('acc')
    print("Total pictures count {}".format(pic_node_num))
    print("Total accounts count {}".format(acc_node_num))
    g.nodes['pic'].data['img_feat'] = torch.ones(pic_node_num, 1000)#
    g.nodes['acc'].data['acc_feat'] = torch.zeros(acc_node_num, 1000)#119typles


    edge_num = g.num_edges('nb')
    print("Total edges count{}".format(edge_num))
    g.edges['nb'].data['label'] = label_torch
    g.edges['nb'].data['train_mask'] = train_mask_torch

    for i in range(pic_node_num):
        video_name = idx2video_name[i]
        feat_offset = i % FRAME_CNT

        video_feature = np.load(os.path.join(feature_path, video_name + ".npy"))
        cur_frame_feature = video_feature[feat_offset]
        cur_frame_feature_th = torch.from_numpy(cur_frame_feature)
        g.nodes['pic'].data['img_feat'][i] = cur_frame_feature_th

    cat_0_set = set()
    cat_1_set = set()
    cat_2_set = set()

    for j in range(acc_node_num):
        account_id = idx2account[j]
        account_info = account2info[account_id]
        fields = account_info.split('_')

        cat_0 = fields[0]
        cat_1 = fields[1]
        cat_2 = fields[2]

        cat_0_set.add(cat_0)
        cat_1_set.add(cat_1)
        cat_2_set.add(cat_2)

        if cat_0 in cat_map:
            g.nodes['acc'].data['acc_feat'][j][cat_map[cat_0]] = 1.0
        if cat_1 in cat_map:
            g.nodes['acc'].data['acc_feat'][j][cat_map[cat_1]] = 1.0
        if cat_2 in cat_map:
            g.nodes['acc'].data['acc_feat'][j][cat_map[cat_2]] = 1.0

    cat_0_list = list(cat_0_set)
    cat_1_list = list(cat_1_set)
    cat_2_list = list(cat_2_set)

    idx = -1
    print("======== Category 0 ========")
    print("A total of {} category 0 found.".format(len(cat_0_list)))
    for i in range(len(cat_0_list)):
        print("\"{}\": {},".format(cat_0_list[i], idx))
        idx += 1
    print("======== Category 1 ========")
    print("A total of {} category 1 found.".format(len(cat_1_list)))
    for i in range(len(cat_1_list)):
        print("\"{}\": {},".format(cat_1_list[i], idx))
        idx += 1
    print("======== Category 2 ========")
    #print("\t".join(cat_2_list))
    for i in range(len(cat_2_list)):
        print("\"{}\": {},".format(cat_2_list[i], idx))
        idx += 1

    #idx2video_name node id -> video  name
    #account2idx

    dgl.save_graphs("./test_data.bin", g)
    #acc_node_num = g.num_nodes('acc')
    #print("Current number of node {} and {}.".format(pic_node_num, acc_node_num))
    #g.nodes['pic'].data['h'] = torch.ones(pic_node_num, 1)
    #g.nodes['acc'].data['h'] = torch.ones(acc_node_num, 1)
    #print(g.successors(0, etype='own'))
    #print(g.ndata['h'][0])
    #print(g.ndata['h'][3])
    #print(g.ndata['h'][10])
    #print(g.num_nodes())


if __name__=="__main__":
    make_graph(sys.argv[1], sys.argv[2])
