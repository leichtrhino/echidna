
import json
from bisect import bisect_right

_datanode_cls_map = dict()
_datanode_name_map = dict()

def get_datanode_cls(name):
    return _datanode_cls_map[name]

def get_datanode_name(cls):
    return _datanode_name_map[cls]

def add_datanode_cls(name, cls):
    _datanode_cls_map[name] = cls
    _datanode_name_map[cls] = name

class DataNode(object):
    def __init__(self, children=None, metrics=None):
        self.parent = None
        self.right_index = None
        self.left_index = None
        self.children = children
        if children:
            for c in children:
                c.parent = self
        self.metrics = metrics

    def init_index(self):
        # set left, right index for all children
        stack = [(self, 0)]
        cur_right_index = 0
        while stack:
            node, phase = stack.pop()
            if phase == 0:
                node.left_index = cur_right_index
                stack.append((node, 1))
                if node.children:
                    for c in node.children[::-1]:
                        stack.append((c, 0))
            elif phase == 1:
                if node.children:
                    assert node.children[-1].right_index is not None
                    node.right_index = node.children[-1].right_index
                else:
                    node.right_index = cur_right_index + 1
                cur_right_index = node.right_index

    def index_not_initialized(self):
        return self.right_index is None or self.left_index is None

    def ensure_index_init(func):
        def wrapper(self, *args, **kwargs):
            if self.index_not_initialized():
                self.init_index()
            return func(self, *args, **kwargs)
        return wrapper

    def cleanup_index(func):
        def wrapper(self, *args, **kwargs):
            retval = func(self, *args, **kwargs)
            self.init_index()
            return retval
        return wrapper

    @ensure_index_init
    def __len__(self):
        return self.right_index - self.left_index

    @ensure_index_init
    def get_single_chain(self, index):
        node = self
        node_list = []
        while node.children:
            # find appropriate child and process
            c_i = bisect_right(
                [c.right_index for c in node.children],
                index
            )
            node = node.children[c_i]

            if node.super_type == 'empty':
                continue
            if not node_list and node.super_type != 'source':
                raise RuntimeError('non source node must be called after '
                                   'source node has been called')
            elif node_list and node.super_type == 'source':
                raise RuntimeError('source node must not be called after '
                                   'source node has been called')

            node_list.append(node)

        return node_list

    @ensure_index_init
    def list_leaf_node(self):
        leaf_node_list = []
        stack = [self]
        while stack:
            node = stack.pop()
            if node.children:
                for c in node.children[::-1]:
                    stack.append(c)
            else: # if not node.children
                leaf_node_list.append(node)

        return leaf_node_list

    @classmethod
    def process_single_chain(cls, node_list):
        # find require_channel_index
        for node in node_list:
            varname = 'channel_index'
            if 'channel_index' not in locals() and node.super_type != 'source':
                raise RuntimeError('non source node must be called after '
                                   'source node has been called')
            if 'channel_index' in locals() and node.super_type == 'source':
                raise RuntimeError('source node must not be called after '
                                   'source node has been called')

            post_req_channel_index = node.require_channel_index_partial()
            if varname not in locals() or channel_index is None:
                channel_index = post_req_channel_index
            elif post_req_channel_index is not None:
                channel_index = [
                    channel_index[pi]
                    for pi in post_req_channel_index
                ]

        if 'channel_index' in locals() and channel_index is not None:
            channel_index = sorted(set(channel_index))
        else:
            channel_index = None

        # chain process
        for node in node_list:
            if node.super_type == 'source':
                if channel_index is None:
                    cur_data = node.process_partial(None)
                else:
                    tmp_data = node.process_partial(channel_index)
                    if len(tmp_data) != len(channel_index):
                        raise RuntimeError('size of required index and '
                                           'return data differ')
                    cur_data = [None for _ in range(len(node.channels))]
                    for i, d in zip(channel_index, tmp_data):
                        cur_data[i] = d

            else: # if node.super_type != 'source':
                cur_data = node.process_partial(cur_data)

        # get metadata
        metadata_list = [
            DataNode.from_dict(node.to_dict_nochildren())
            for node in node_list
        ]

        return cur_data, metadata_list

    @ensure_index_init
    def process(self, index):
        node_list = self.get_single_chain(index)
        return DataNode.process_single_chain(node_list)

    def __getitem__(self, index):
        return self.process(index)

    @property
    def super_type(self):
        raise NotImplementedError()

    def process_partial(self):
        raise NotImplementedError()

    def require_channel_index_partial(self):
        raise NotImplementedError()

    def push_metric(self, key, value):
        if self.super_type == 'empty':
            raise ValueError('metric cannot be pushed to an empty node')
        if not hasattr(self, 'metrics') or self.metrics is None:
            self.metrics = dict()
        self.metrics[key] = value

    def get_metric(self, key):
        if not hasattr(self, 'metrics') or self.metrics is None:
            return None
        return self.metrics.get(key)

    @classmethod
    def from_dict(cls,
                  obj : dict,
                  context : dict=dict()):
        datanode_cls = get_datanode_cls(obj['type'])
        datanode_obj = datanode_cls.from_dict_args(obj['args'], context)
        if obj.get('children'):
            datanode_obj.children = [
                cls.from_dict(c, context) for c in obj.get('children')
            ]
            for c in datanode_obj.children:
                c.parent = datanode_obj
        if obj.get('metrics'):
            datanode_obj.metrics = obj.get('metrics')
        return datanode_obj

    def from_dict_args(cls,
                       obj : dict,
                       context : dict=None):
        raise NotImplementedError()

    def to_dict(self):
        return {
            'type': get_datanode_name(type(self)),
            'args': self.to_dict_args(),
        } | ({
            'children': [
                c.to_dict() for c in self.children
            ]
        } if hasattr(self, 'children') and self.children else {}) | ({
            'metrics': self.metrics
        } if hasattr(self, 'metrics') and self.metrics else {})

    def to_dict_args(self):
        raise NotImplementedError()

    def to_dict_nochildren(self):
        return {
            'type': get_datanode_name(type(self)),
            'args': self.to_dict_args(),
        }

    @cleanup_index
    def balance_by_remove(self):
        node_list = []
        depth_map = dict()
        queue = [(self, 0)]
        while queue:
            node, depth = queue.pop(0)
            node_list.append(node)
            depth_map[node] = depth
            if node.children:
                for c in node.children:
                    queue.append((c, depth+1))

        # NOTE: this tree is traversed by BFS, so node_list is
        #       ascending order in meaning of depth of node
        for n in node_list[::-1]:
            if n.parent:
                depth_map[n.parent] = max(
                    depth_map[n], depth_map[n.parent])

        stack = [self]
        max_depth = depth_map[self]
        while stack:
            node = stack.pop()
            if node.children:
                node.children = [
                    c for c in node.children if depth_map[c] == max_depth]
                for c in node.children:
                    stack.append(c)


class EmptyNode(DataNode):
    @property
    def super_type(self):
        return 'empty'

    @classmethod
    def from_dict_args(cls,
                       obj : dict,
                       context : dict=None):
        return cls()

    def to_dict_args(self):
        return None

add_datanode_cls('empty', EmptyNode)

