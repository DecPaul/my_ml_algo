class Node(object):
    def __init__(self, depth):
        pass

# 树的叶子节点，对于N分类问题，value是一个长度为N的数组，存储每个类别的概率；
# 对于回归问题，value是一个浮点数，存储该节点上所有样本的y均值
class LeafNode(Node):
    def __init__(self, value):
        self.value = value

# 树的中间节点，feature存储用于分裂该节点的特征的id，threshold存储切分阈值
class MiddleNode(Node):
    def __init__(self, left, right, feature, threshold):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold