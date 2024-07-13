from collections import deque, defaultdict
from functools import cache
import re
from random import choice
from random import seed as rseed

_newick_internal=re.compile(r'\((.+),\s*(.+)\)(.*):([.\d]+);*')
_newick_leaft=re.compile(r'(.+):([.\d]+);*')

class TreeNode:

    __slots__ = ("_id", "_left", "_right", "_ancestor", "_branchlen", "_is_outgroup")

    def __init__(self, id: int, branchlen: float = 1.0, is_outgroup=False) -> None:
        self._id = id
        self._branchlen = branchlen
        self._is_outgroup = is_outgroup

    @property
    def is_outgroup(self):
        return self._is_outgroup

    @property
    def branchlen(self):
        return self._branchlen

    @branchlen.setter
    def branchlen(self, branchlen : float):
        self._branchlen = branchlen

    def copy_fwd(self):
        cp = TreeNode(self.id, self.branchlen)
        if not self.is_leaf:
            cp.set_children(self.left.copy_fwd(), self.right.copy_fwd())
        return cp
    
    def copy_rev(self):
        cp = TreeNode(self.id, self.branchlen)
        if not self.is_root:
            sibling = TreeNode(self.sibling.id, self.sibling.branchlen)
            self.ancestor.copy_rev().set_children(cp, sibling)
        return cp

    def ribs(self):
        '''
        Ribs are the siblings of the ancestors of a node.
        '''
        currnode=self
        while not currnode.is_root:
            yield currnode.sibling
            currnode = currnode.ancestor

    def _set_ancestor(self, node):
        self._ancestor = node

    ##
    # Trees can only be built in a top-down manner so that the ancestor/child relationship is maintained
    # Trees can only be modified by severing both children from a node so that each subtree is a valid binary tree,
    #   and no node is left with only one child.
    ##
    def set_children(self, left, right):
        if hasattr(self, "_left") or hasattr(self, "_right"):
            raise ValueError("Node already has children, you must sever them first")

        assert left is not right, "Left and right children must be distinct"
        self._left = left
        left._set_ancestor(self)
        self._right = right
        right._set_ancestor(self)

    def sever_children(self):
        left = self._left
        right = self._right

        del left._ancestor
        del self._left
        del right._ancestor
        del self._right

        return (left, right)

    @property
    def id(self):
        return self._id

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def ancestor(self):
        return self._ancestor

    @property
    def children(self):
        return self.left, self.right

    @property
    def is_root(self) -> bool:
        return not hasattr(self, "_ancestor")

    @property
    def is_leaf(self) -> bool:
        return not hasattr(self, "_left") or not hasattr(self, "_right")

    @property
    def sibling(self):
        if self.is_root:
            raise ValueError("Root has no ancestor")

        try:
            return (
                self.ancestor.left
                if self.ancestor.right == self
                else self.ancestor.right
            )
        except AttributeError as err:
            raise ValueError("Root has no sibling") from err

    def __repr__(self):
        return f"TreeNode({self.id})"

    def __str__(self):
        return f"TreeNode({self.id})"

    def bfs_nodes(self):
        q = deque([self])
        while q:
            node = q.popleft()
            yield node
            if not node.is_leaf:
                q.extend(node.children)

    def dfs_nodes_rootorder(self):
        stack = deque([self])
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf:
                stack.extend(node.children)

    def trace_ancestors(self):
        currnode = self
        while not currnode.is_root:
            currnode = currnode.ancestor
            yield currnode

    def jump_to_root(self):
        currnode = self
        while not currnode.is_root:
            currnode = currnode.ancestor
        return currnode

    def get_nni_neighborhoold(self):
        """
        Returns the nodes involved in the NNI neighborhood of the current node.
        The neighborhood consists of the current node, its children, its sibling, its ancestor, and its ancestor's sibling.
        Using the following layout:
        
          K
         /
        I   M
         \ /
          J
           \
            L
        
        This function returns (L, M, J, K, I).
        """
        assert (
            not self.is_leaf or self.is_root
        ), "Cannot perform NNI on a leaf node or root node"
        return (self.left, self.right, self, self.sibling, self.ancestor)


    def nearest_neighbor_interchange(self, about: str) -> None:
        """
        Uses the following layout:
        
          K
         /
        I   M
         \ /
          J
           \
            L
        
        Where J* is the node on which the NNI is performed.
        The "about" argument specifies whether the NNI is performed on the left or right child of J*.
        """

        if about not in ("left", "right"):
            raise ValueError('Argument "about" must be either "left" or "right"')

        assert (
            not self.is_leaf or self.is_root
        ), "Cannot perform NNI on a leaf node or root node"

        I = self.ancestor
        K = self.sibling

        # break current connections
        L, M = self.sever_children()
        I.sever_children()

        # reconnect
        if about == "right":
            I.set_children(self, M)
            self.set_children(L, K)
        else:
            I.set_children(self, L)
            self.set_children(K, M)

    def draw(self, depth=0, fromlevel=0):
        print(" " * 4 * fromlevel + "|" + "-" * (depth - fromlevel) * 4 + str(self.id))
        if not self.is_leaf:
            self.right.draw(depth + 1, fromlevel=depth)
            self.left.draw(depth + 1, fromlevel=depth)

    def get_downstream_leaves(self):
        if self.is_leaf:
            return [self]
        else:
            return (
                self.left.get_downstream_leaves() + self.right.get_downstream_leaves()
            )
        
    def get_distance_from_leaves(self):
        if self.is_leaf:
            return 0
        else:
            return 1 + max(self.left.get_distance_from_leaves(), self.right.get_distance_from_leaves())
    


class BinaryTree:
    """
    A functional wrapper around the TreeNode class,
    most methods are static and take a TreeNode object as the first argument.
    """

    @staticmethod
    def list_distances_from_leaves(root: TreeNode):
        '''
        This trick with cache is used to memoize the results of the get_distance_from_leaves method
        so that repeated recursive calls to this function are not necessary.

        The cache is cleared after this function is called so that modifications to the tree
        are not ignored for the cached values.
        '''
        leaf_dist = defaultdict(list)
        df=cache(TreeNode.get_distance_from_leaves)
        
        for node in BinaryTree.dfs_nodes_rootorder(root):
            leaf_dist[df(node)].append(node)
        
        return leaf_dist


    @staticmethod
    def fetch_node(root: TreeNode, id: int):
        '''
        Unfortunately, the tree is dynamic, so we can't just index into pre-compiled list of nodes.
        This function will search the tree for a node with the given id.
        '''
        for node in BinaryTree.dfs_nodes_rootorder(root):
            if node.id == id:
                return node
        raise ValueError(f"Node with id {id} not found in tree")

    @staticmethod
    def _make_tree(node: TreeNode, n_levels: int):
        if n_levels == 0:
            return

        left = TreeNode(node.id * 2 + 1)
        right = TreeNode(node.id * 2 + 2)
        node.set_children(left, right)
        BinaryTree._make_tree(left, n_levels - 1)
        BinaryTree._make_tree(right, n_levels - 1)


    @staticmethod
    def make_tree(n_levels : int):
        root = TreeNode(0)
        BinaryTree._make_tree(root, n_levels)
        return root

    @staticmethod
    def copy_fwd(node: TreeNode):
        return node.copy_fwd()
    
    @staticmethod
    def copy_rev(node: TreeNode):
        return node.copy_rev()

    @staticmethod
    def list_nodes(root: TreeNode):
        return list(root.dfs_nodes_rootorder())[::-1]

    @classmethod
    def list_edges(cls, root: TreeNode):
        return [
            (node.ancestor, node) for node in cls.list_nodes(root) if not node.is_root
        ]

    @staticmethod
    def dfs_nodes_rootorder(root: TreeNode):
        return root.dfs_nodes_rootorder()

    @staticmethod
    def bfs_nodes(root: TreeNode):
        return root.bfs_nodes()

    @staticmethod
    def draw(root: TreeNode):
        return root.draw()

    @staticmethod
    def get_downstream_leaves(root: TreeNode):
        return root.get_downstream_leaves()

    @staticmethod
    def nearest_neighbor_interchange(node: TreeNode, about: str):
        return node.nearest_neighbor_interchange(about)

    @staticmethod
    def get_nni_neighborhoold(node: TreeNode):
        return node.get_nni_neighborhoold()

    @staticmethod
    def trace_ancestors(node: TreeNode):
        return node.trace_ancestors()

    @staticmethod
    def to_newick(root: TreeNode):
        
        def _recurse_newick(node):
            if node.is_leaf:
                return f"{node.id}:{node.branchlen}"
            else:
                return f"({_recurse_newick(node.left)}, {_recurse_newick(node.right)}):{node.branchlen}"

        return _recurse_newick(root) + ";"
    

    @staticmethod
    def from_newick(newick_str: str):
        
        def _parse_newick(newick_str, id):
            if _newick_internal.match(newick_str):
                left, right, name, branchlen = _newick_internal.match(newick_str).groups()
                name = str(id) if name=='' else name
                node = TreeNode(name, float(branchlen))
                node.set_children(_parse_newick(left, 2*id+1), _parse_newick(right, 2*id+2))
                return node
            elif _newick_leaft.match(newick_str):
                name, branchlen = _newick_leaft.match(newick_str).groups()
                return TreeNode(name, float(branchlen))
            else:
                raise ValueError("Invalid Newick string: {}".format(newick_str))
            
        return _parse_newick(newick_str, 0)


    @staticmethod
    def random_tree(leaf_ids, 
                    outgroup=None, 
                    seed=1776,
                    branchlen_generator=lambda: 1.0):

        rseed(seed)
        nodes_container = list(TreeNode(id, branchlen=branchlen_generator()) for id in leaf_ids)
        n_merges=0
        while len(nodes_container) > 1:
            n_merges+=1
            left = choice(nodes_container)
            nodes_container.remove(left)

            right = choice(nodes_container)
            nodes_container.remove(right)

            parent = TreeNode(f'Inner{n_merges}', branchlen=branchlen_generator())
            parent.set_children(left, right)
            nodes_container.append(parent)

        current_root = nodes_container.pop(0)

        if not outgroup is None:
            newroot = TreeNode('Root')
            newroot.set_children(
                current_root, 
                TreeNode(outgroup, is_outgroup=True, branchlen=branchlen_generator())
            )
            return newroot
        else:
            return current_root
    


