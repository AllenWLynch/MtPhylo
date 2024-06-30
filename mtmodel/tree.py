import networkx as nx
from numpy import array
from copy import deepcopy

## A simple abstraction of the graph logic from the math
## Implement a new interface if needed

class NoSiblingError(Exception):
    pass

def get_root(tree):
    return tree.graph['root']

def get_ancestor(tree, node):
    return next(tree.predecessors(node))

def get_ancestor_branch_length(tree, node):
    anc=next(tree.predecessors(node))
    return tree.edges[anc, node]['branch_length']

def get_descendents(tree, node):
    return list(tree.successors(node))

def get_sibling(tree, node):
    anc=next(tree.predecessors(node))
    sibs=[s for s in tree.successors(anc) if not s == node]
    if len(sibs) > 1:
        raise ValueError(f'More than one sibling for node {node} found. Tree is not binary.')
    elif len(sibs) == 0:
        raise NoSiblingError(f'No sibling for node {node} found. Tree is not binary')
    return sibs[0]


def get_edges_dfs(tree, root_node):
    return list(nx.dfs_edges(tree, root_node))

def get_nodes_dfs(tree, root_node):
    return list(nx.dfs_postorder_nodes(tree, root_node))

def set_branchlen(tree, edge, len):
    tree.edges[edge]['branch_length'] = len

def get_branchlen(tree, edge):
    return tree.edges[edge]['branch_length']

def get_leaves(tree):
    return [node for node in tree.nodes if tree.out_degree(node) == 0]

def get_branchlen_vector(tree, root_node=None):
    root_node=get_root(tree) if root_node is None else root_node
    return array([get_branchlen(tree, edge) for edge in get_edges_dfs(tree, root_node)])


def get_NNI_1(ctree, edge):

    #copied_edge = next(edge for edge in ctree.edges if (edge[0].name, edge[1].name) == edge_key)

    parent, child = edge
    sibling = get_sibling(ctree, child)
    (left, right) = get_descendents(ctree, child)

    bl1 = get_branchlen(ctree, (child, right)); bl2 = get_branchlen(ctree, (parent, sibling))
    ctree.remove_edge(child, right)
    ctree.remove_edge(parent, sibling)
    ctree.add_edge(parent, right, branch_length=bl1)
    ctree.add_edge(child, sibling, branch_length=bl2)

    optim_nodes = (sibling, left, child, right, parent)
    optim_edges = list(set(nx.dfs_edges(ctree, parent, 1)).union(nx.dfs_edges(ctree, child, 1)))

    blacklist_edges = [(parent, child), (child, right), (parent, sibling), (child, left)]

    return ctree, optim_edges, optim_nodes, blacklist_edges


def get_NNI_2(ctree, edge):

    #copied_edge = next(edge for edge in ctree.edges if (edge[0].name, edge[1].name) == edge_key)

    parent, child = edge
    sibling = get_sibling(ctree, child)
    (left, right) = get_descendents(ctree, child)

    bl1 = get_branchlen(ctree, (child, left)); bl2 = get_branchlen(ctree, (parent, sibling))
    ctree.remove_edge(child, left)
    ctree.remove_edge(parent, sibling)
    ctree.add_edge(parent, left, branch_length=bl1)
    ctree.add_edge(child, sibling, branch_length=bl2)

    optim_nodes = (sibling, right, child, left, parent)
    optim_edges = list(set(nx.dfs_edges(ctree, parent, 1)).union(nx.dfs_edges(ctree, child, 1)))

    blacklist_edges = [(parent, child), (child, right), (parent, sibling), (child, left)]

    return ctree, optim_edges, optim_nodes, blacklist_edges



def to_newick(g):

    def _iter_newick(g, node):
        subgs = []
        for child in g[node]:
            if len(g[child]) > 0:
                subgs.append(_iter_newick(g, child) + f'{child.name}:{g.edges[node, child]["branch_length"]}')
            else:
                subgs.append(f'{child.name}:{g.edges[node, child]["branch_length"]}')
        
        return "(" + ','.join(subgs) + ")"

    return _iter_newick(g, g.graph['root']) + str(g.graph['root'].name) + ':0.;'