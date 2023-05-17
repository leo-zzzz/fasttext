import re
from types import MethodType, FunctionType
from random import shuffle
import pandas as pd
import jieba
import numpy as np
import os
import csv
import fasttext
import fasttext.util
import xlrd
import xlwt


def clean_txt(raw):
    # fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    fil = re.compile(r"[^0-9\u4e00-\u9fa5]+")
    return fil.sub(' ', raw)

def seg(sentence, sw, apply=None):
    if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
        sentence = apply(sentence)
    return ' '.join([i for i in jieba.cut(sentence) if i.strip() and i not in sw])

def stop_words():
    with open('data/stopwords.txt', 'r', encoding='utf-8') as swf:
        return [line.strip() for line in swf]

class _MD(object):
    mapper = {
        str: '',
        int: 0,
        list: list,
        dict: dict,
        set: set,
        bool: False,
        float: .0
    }

    def __init__(self, obj, default=None):
        self.dict = {}
        assert obj in self.mapper, \
            'got a error type'
        self.t = obj
        if default is None:
            return
        assert isinstance(default, obj), \
            f'default ({default}) must be {obj}'
        self.v = default

    def __setitem__(self, key, value):
        self.dict[key] = value


    def __getitem__(self, item):
        if item not in self.dict and hasattr(self, 'v'):
            self.dict[item] = self.v
            return self.v
        elif item not in self.dict:
            if callable(self.mapper[self.t]):
                self.dict[item] = self.mapper[self.t]()
            else:
                self.dict[item] = self.mapper[self.t]
            return self.dict[item]
        return self.dict[item]

def defaultdict(obj, default=None):
    return _MD(obj, default)

class TransformData(object):
    def to_csv(self, handler, output, index=False):
        dd = defaultdict(list)
        for line in handler:
            label, content = line.split(',', 1)
            # print(content)
            dd[label.strip('__label__').strip()].append( seg(content.lower().replace('\n', ''), stop_words(), apply=clean_txt))

        df = pd.DataFrame()
        for key in dd.dict:
            col = pd.Series(dd[key], name=key)
            df = pd.concat([df, col], axis=1)
        return df.to_csv(output, index=index, encoding='utf-8')

def split_train_test(source, auth_data=False):
    if not auth_data:
        train_proportion = 0.8
    else:
        train_proportion = 0.98

    basename = source.rsplit('.', 1)[0]
    train_file = basename + '_train.txt'
    test_file = basename + '_test.txt'

    handel = pd.read_csv(source, index_col=False, low_memory=False)
    train_data_set = []
    test_data_set = []
    for head in list(handel.head()):
        train_num = int(handel[head].dropna().__len__() * train_proportion)
        sub_list = [f'__label__{head} , {item.strip()}\n' for item in handel[head].dropna().tolist()]
        train_data_set.extend(sub_list[:train_num])
        test_data_set.extend(sub_list[train_num:])
    shuffle(train_data_set)
    shuffle(test_data_set)

    with open(train_file, 'w', encoding='utf-8') as trainf,\
        open(test_file, 'w', encoding='utf-8') as testf:
        for tds in train_data_set:
            trainf.write(tds)
        for i in test_data_set:
            testf.write(i)

    return train_file, test_file

def train_model(ipt=None, opt=None, model='', dim=None, epoch=None, lr=None, loss='softmax'):
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(
                        ipt,
                        label = '__label__',
                        dim = dim,
                        epoch = epoch,
                        lr = lr,
                        # minCount=100,
                        loss = loss,
                        # minCountLabel = 300,
                        minn = 2,
                        maxn = 4,
                        neg = 5,               
                        wordNgrams = 3,
                        # pretrainedVectors='./data/cc.zh.300.vec',
                        lrUpdateRate = 50,      
                        t = 0.0001
                        )
        """
          训练一个监督模型, 返回一个模型对象
          
          @param input:           训练数据文件路径
          @param lr:              学习率
          @param dim:             向量维度
          @param ws:              cbow模型时使用
          @param epoch:           次数
          @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
          @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
          @param minn:            构造subword时最小char个数
          @param maxn:            构造subword时最大char个数
          @param neg:             负采样
          @param wordNgrams:      n-gram个数
          @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
          @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
          @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
          @param lrUpdateRate:    学习率更新
          @param t:               负采样阈值
          @param label:           类别前缀
          @param verbose:         ??
          @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
          @return model object
        """
        classifier.save_model(opt)
    return classifier


# 把字典形式的数据写入csv文件，计算标签的准确率和f1
def label_PreF1(classifier, test_file):
    result = classifier.test_label(test_file, k=3, threshold=0.4)
    file = open('result.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=['标签', '准确率', '召回率', 'f1'])
    csv_writer.writeheader()
    key = list(result.keys())
    # 取出values集合,或者直接根据 字典的key值读取。
    value = list(result.values())
    for i in range(len(key)):
        dic = {  # 字典类型
            '标签': key[i].split('__label__')[1],
            '准确率': value[i].get('precision'),
            '召回率': value[i].get('recall'),
            'f1': value[i].get('f1score'),
        }
        csv_writer.writerow(dic)  # 数据写入csv文件
    file.close()

    # 计算所有标签的准确率，f1的平均值
    with open('result.csv', encoding='utf-8') as csv_file:
        row = csv.reader(csv_file, delimiter=',')
        next(row)  # 读取首行
        price1 = []
        price2 = []
        # 读取除首行之后每一行的第二列数据，并将其加入到数组price之中
        for r in row:
            price1.append(float(r[1]))  # 将字符串数据转化为浮点型加入到数组之中
            price2.append(float(r[3]))  # 将字符串数据转化为浮点型加入到数组之中
    print('准确率：', np.mean(price1))
    print('f1：', np.mean(price2))


def cal_precision_and_recall(classifier,file=''):
    precision = defaultdict(int, 1)
    recall = defaultdict(int, 1)
    total = defaultdict(int, 1)
    with open(file,encoding='utf-8') as f:
        for line in f:
            label, content = line.split(',', 1)
            total[label.strip().strip('__label__')] += 1
            # print(total.dict)
            labels2 = classifier.predict([seg(sentence=content.strip(), sw=stop_words(), apply=clean_txt)])
            # print(labels2)  # labels2格式是这样([['__label__机械结构强度学']], [array([0.7304502], dtype=float32)])
            pre_label, sim = labels2[0][0][0], labels2[1][0][0]  # 拿出  __label__机械结构强度   0.7304502
            recall[pre_label.strip().strip('__label__')] += 1

            if label.strip() == pre_label.strip():
                precision[label.strip().strip('__label__')] += 1

    # print('precision', precision.dict)
    # print('recall', recall.dict)
    # print('total', total.dict)
    f_result=open('each_result.txt','w', encoding='utf-8')
    for sub in precision.dict:
        pre = precision[sub] / total[sub]
        rec =  precision[sub] / recall[sub]
        F1 = (2 * pre * rec) / (pre + rec)
        # print(f"{sub.strip('__label__')}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")
        f_result.write(f"{sub.strip('__label__')}, {str(pre)}, {str(rec)}, {str(F1)}\n")


def main(source):
    basename = source.rsplit('.', 1)[0]
    csv_file = basename + '.csv'

    # td = TransformData()
    # handler = open(source,encoding='UTF-8')
    # td.to_csv(handler, csv_file)
    # handler.close()

    train_file, test_file = split_train_test(csv_file)
    dim = 100
    lr = 0.05
    epoch = 100
    
    model = f'data2/data_dim{str(dim)}_lr{str(lr)}_iter{str(epoch)}.model'

    classifier = train_model(ipt=train_file,
                             opt=model,
                             model=model,
                             dim=dim, epoch=epoch, lr=lr)

    # 验证测试集效果
    result = classifier.test_label(test_file, k=3, threshold=0.4)
    f_result=open('result.txt','w',encoding='utf-8')
    f_result.write(str(result))

    label_PreF1(classifier, test_file)

    cal_precision_and_recall(classifier,test_file)


    # 对csv文件开始做预测
    '''f_write_pred_data=open('data2/project1_pred.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f_write_pred_data)
    # header=['项目名称','预测label','概率']
    # csv_writer.writerow(header)
    with open('data2/project1.csv', encoding='gb18030') as f:
        for row in csv.reader(f, skipinitialspace=True):
            data = []
            item_name = str(row[1]) + str(row[12])
            deal_label_for_pred = seg(str(item_name).lower().replace('\n', ''), stop_words(), apply=clean_txt)
            pred_label = classifier.predict(str(deal_label_for_pred))
            # print(pred_label, pred_label[0])
            # data.append(str(row[0]))
            data.append(str(row[1]))
            # data.append(str(row[2]))
            # data.append(str(row[3]))
            # data.append(str(row[4]))
            data.append(str(row[12]))
            # data.append(str(row[6]))
            # data.append(str(row[7]))
            # data.append(str(row[8]))
            # data.append(str(row[9]))
            # data.append(str(row[10]))
            # data.append(str(row[11]))
            # data.append(str(row[12]))
            # data.append(str(row[13]))
            # data.append(str(row[14]))
            # data.append(str(row[15]))
            # data.append(str(row[16]))
            # data.append(str(row[17]))
            data.append(str(pred_label[0]).split('__label__')[1].split("'")[0])
            data.append((str(pred_label[1]).split('[')[1].split(']')[0]))
            csv_writer.writerow(data)'''


    # 对xlsx文件做预测
    '''file_name="data2/patent.xlsx"
    open_csv=xlrd.open_workbook(file_name)
    pred_data = open('data2/patent_pred.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(pred_data)
    ws = open_csv.sheet_by_name("patent")
    for i in range(ws.nrows):
        data=[]
        # .cell_value
        item_name=str(ws.cell_value(i,1))+str(ws.cell_value(i,5))
        # print(item_name)
        deal_label_for_pred=seg(str(item_name).lower().replace('\n', ''), stop_words(), apply=clean_txt)
        pred_label=classifier.predict(str(deal_label_for_pred))
        # data.append(str(ws.cell_value(i,0)))
        data.append(str(ws.cell_value(i,1)))
        # data.append(str(ws.cell_value(i,2)))
        # data.append(str(ws.cell_value(i,3)))
        # data.append(str(ws.cell_value(i,4)))
        # data.append(str(ws.cell_value(i,5)))
        # data.append(str(ws.cell_value(i,6)))
        # data.append(str(ws.cell_value(i,7)))
        # data.append(str(ws.cell_value(i,8)))
        data.append(str(ws.cell_value(i,5)))
        # data.append(i)
        # data.append(str(true_label))
        data.append(str(pred_label[0]).split('__label__')[1].split("'")[0])
        # data.append(str(pred_label[1]).split('[')[1].split(']')[0])
        csv_writer.writerow(data)'''

    # 读取xlsx文件添加预测列后并写回该xlsx文件
    # 打开工作簿
    workbook = xlrd.open_workbook("./data2/pre_test01.xlsx")
    # 获取第一个工作表
    worksheet = workbook.sheet_by_index(0)
    # 创建一个新的工作簿
    workbook_new = xlwt.Workbook()
    # 创建一个新的工作表
    worksheet_new = workbook_new.add_sheet('Sheet2')
    # 遍历每一行
    for i in range(worksheet.nrows):
        # 获取当前行的数据
        row_data = worksheet.row_values(i)
        item_name = str(worksheet.cell_value(i, 1)) + str(worksheet.cell_value(i, 2))
        deal_label_for_pred = seg(str(item_name).lower().replace('\n', ''), stop_words(), apply=clean_txt)
        pred_label = classifier.predict(str(deal_label_for_pred))
        # 在当前行的最后添加一个新单元格，并将数据写入该单元格
        row_data.append(str(pred_label[0]).split('__label__')[1].split("'")[0])
        # 遍历每一列，并将数据写入新单元格
        for col, data in enumerate(row_data):
            worksheet_new.write(i, col, data)
    # 保存新的工作簿
    workbook_new.save("./data2/pre_test01.xlsx")


    '''f_read_pred_data=open('../QingHai_data/project1.txt','r',encoding='utf-8')
    f_write_pred_data=open('data/project1_pred_unsupervied.csv','a',newline='')
    csv_writer = csv.writer(f_write_pred_data)
    header=['项目名称','预测label','概率']
    csv_writer.writerow(header)
    # 项目的真实label
    for line in f_read_pred_data.readlines():
        data=[]
        # true_label=line.split(',',1)[0]
        # item_name=line.split(',',1)[0]
        item_name=line
        # true_name=line.split(',',1)[1]
        # print(true_name)
        deal_label_for_pred=seg(str(item_name).lower().replace('\n', ''), stop_words(), apply=clean_txt)
        pred_label=classifier.predict(str(deal_label_for_pred))
        data.append(str(item_name))
        # data.append(str(true_label))
        data.append(str(pred_label[0]).split('__label__')[1].split("'")[0])
        data.append(str(pred_label[1]).split('[')[1].split(']')[0])
        csv_writer.writerow(data)'''



if __name__ == '__main__':
    main('data2/data2.txt')
    # item_name="hello computer 你是计算机123"
    # print(seg(str(item_name).lower().replace('\n', ''), stop_words(), apply=clean_txt))
