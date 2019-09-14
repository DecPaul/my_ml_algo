import numpy as np
from tree import *
from loss import *

# 手写一个简易的树模型训练工具
# 本工具支持训练分类模型和回归模型，分类模型支持多分类。
# 输入特征支持连续型浮点特征和离散型特征

class DT(object):
    '''
    模型初始化参数说明：
    1 classifier：标识分类器还是回归器，True是分类，False是回归
    2 max_depth：节点深度在达到max_depth时停止继续分裂
    3 min_samples_split：当某个节点的样本数少于min_samples_split时，就不分裂了
    4 max_features: 每次节点分裂时需要使用max_features个特征，默认使用全部特征
    5 criterion：节点分裂时的计算标准。目前分类树支持信息熵(entropy)和基尼系数(gini)，回归树仅支持mse
    '''
    def __init__(self, classifier=True, max_depth=None, min_samples_split=2,
                 max_features=None, criterion='entropy', seed=None):
        if seed:
            np.random.seed(seed)

        self.root = None

        self.classifier = classifier
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion

        if classifier and criterion == 'mse':
            raise Exception('classify model cannot use mse')
        if not classifier and (criterion == 'gini' or criterion=='entropy'):
            raise Exception('regression model cannot use gini or entropy')

    def fit(self, X, y):
        '''
        模型训练主调函数
        :param X: N * M 的 numpy 二维数组，N 是样本个数，M 是特征个数
        :param y: N 的 numpy 一维数组
        使用此工具训练的样本预处理说明：
        1 同一个特征需要用相同的数据类型，不能有空值。
        2 因为此树是一颗二叉树，所以对于多类别的特征，如果特征值无序，请自行转成one-hot形式再输入
        3 对多分类问题，y 请自行转成 0 1 2 3 4 ... ; 对回归问题，y 扔进来浮点数就行
        '''
        if X.shape[0] == 0 or len(y) == 0 or X.shape[0] != len(y):
            raise Exception('check your length of X or y')
        self.root = self._build_tree(X, y, 1)

    def predict(self, X):
        '''
        模型预测主调函数
        :param X: N * M 的 numpy 二维数组，N 是样本个数，M 是特征个数
        :return: 对每个样本，分类问题返回各个类别的概率，回归问题返回一个浮点数
        '''
        return np.array([self._traverse(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        '''
        自上而下递归建树，depth维护构造的节点的深度
        '''
        if len(set(y)) == 1:
            # 此节点的y都相同，不需要继续分裂了
            if self.classifier:
                values = np.zeros(self.class_num)
                values[y[0]] = 1
                return LeafNode(values)
            else:
                return LeafNode(y[0])

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            # 达到用户指定的最大深度或节点内样本数少于用户指定数，则不分裂
            if self.classifier:
                values = np.bincount(y, minlength=self.class_num) / len(y)
                return LeafNode(values)
            else:
                return LeafNode(np.mean(y))

        N, M = X.shape
        child_depth = depth + 1
        feature_idxs = np.random.choice(M, self.max_features, replace=False) # 从[0,M)中无放回采样

        # 遍历每个特征，选出最优的分裂特征和分裂阈值
        feature, threshold = self._break(X, y, feature_idxs)

        # np.argwhere()函数返回所有满足条件的索引数组
        left_idx_array = np.argwhere(X[:, feature] <= threshold).flatten()
        right_idx_array = np.argwhere(X[:, feature] > threshold).flatten()

        left = self._build_tree(X[left_idx_array, :], child_depth)
        right = self._build_tree(X[right_idx_array, :], child_depth)

        return MiddleNode(left, right, feature, threshold)

    def _break(self, X, y, feature_idxs):
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        for i in feature_idxs:
            col_data = X[:, i]
            levels = np.unique(col_data) # np.unique()函数返回一个数组排序后去重的结果
            # 交错相加取均值，比如[1,2,3,4,5] -> [1.5, 2.5, 3.5, 4.5] ，作为候选的阈值
            thresholds = (levels[:-1] + levels[1:]) / 2
            max_gain = (None, -np.inf)
            for t in thresholds:
                cur_gain = self._info_gain(y, t, col_data)
                if cur_gain > max_gain:
                    max_gain = (t, cur_gain)
            if max_gain[1] > best_gain:
                best_gain = max_gain[1]
                best_feature_idx = i
                best_threshold = max_gain[0]
        return best_feature_idx, best_threshold


    def _info_gain(self, y, threshold, col_data):
        # 计算当前节点上的所有样本，在当前特征的当前分割点下，所选择的criterion的信息增益
        if self.criterion == 'mse':
            loss = mse
        elif self.criterion == 'entropy':
            loss = entropy
        elif self.criterion == 'gini':
            loss = gini

        parent_loss = loss(y)

        left_idx_array = np.argwhere(col_data <= threshold).flatten()
        right_idx_array = np.argwhere(col_data > threshold).flatten()

        # 计算子节点的loss的时候，简单按照样本个数比例做加权，惩罚一下样本数太少的子节点
        left_num = len(left_idx_array)
        right_num = len(right_idx_array)
        n = left_num + right_num
        left_loss = loss(y[left_idx_array])
        right_loss = loss(y[right_idx_array])

        child_loss = (left_num / n) * left_loss + (right_num / n) * right_loss

        return parent_loss - child_loss

    def _traverse(self, x, node):
        # 对一个样本，遍历树得到它应在的叶子节点，返回它的预测label
        if isinstance(node, LeafNode):
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
