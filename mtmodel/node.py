from .tree import *
import numpy as np
from scipy.optimize import minimize, Bounds
import warnings
import logging
from numba import njit
logger = logging.getLogger('BranchlenOptimizer')


def log_safe_matmul(T, log_A):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        alpha = np.max(log_A, axis=0, keepdims=True)
        return np.nan_to_num(
            np.log(T @ np.exp(log_A - alpha)) + alpha,
            nan=-np.inf
        )

def log_safe_vecmul(V, log_A):
    alpha = np.max(log_A, axis=0)
    return np.nan_to_num(np.log(V @ np.exp(log_A - alpha)) + alpha, nan=-np.inf)


@njit('float64[:,:](float64[:,:], float64[:,:])', cache=True)
def log_safe_matmul(T, log_A):
    alpha = np.max(log_A)
    return np.nan_to_num(
        np.log(T @ np.exp(log_A - alpha)) + alpha,
        nan=-np.inf
    )
    

class Node:
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class LeafNode(Node):

    def __init__(self, 
                 name : str, 
                 model,
                 observation : np.ndarray,
                 ) -> None:
        self.name = name
        self.model = model

        self.L_ = model.logp_obs_given_state(
                        model.add_constant_site(observation)
                    )

        self.A_ = np.zeros_like(self.L_)
        self.T_ = np.zeros((model.n_states, model.n_states))


    def backward(self, tree):
        ancestor_branch_length = get_ancestor_branch_length(tree, self)
        self.T_[...] = self.model.transition_matrix(ancestor_branch_length)
        self.A_[...] = log_safe_matmul(self.T_, self.L_) #(N_states, N_sites)


    def forward(self, tree):
        pass


class InternalNode(Node):

    def __init__(self, 
                 name : str, 
                 model,
                 ) -> None:
        
        self.name = name
        self.model = model
        
        self.L_ = np.zeros((model.n_states, model.site_dim))
        self.A_ = np.zeros_like(self.L_)
        self.B_ = np.zeros_like(self.L_)
        self.T_ = np.zeros((model.n_states, model.n_states))


    def forward(self,tree):

        ancestor = get_ancestor(tree, self)
        try:
            A = get_sibling(tree, self).A_ # (N_states, N_sites)
        except NoSiblingError:
            A = 0.

        B = ancestor.B_ # (N_states, N_sites)
        T = self.T_ # (N_states, N_states)

        self.B_[...] = log_safe_matmul(T.T, A + B) # (N_states, N_sites)


    def backward(self,tree):

        ancestor_branch_length = get_ancestor_branch_length(tree, self)
        descendents = get_descendents(tree, self)

        self.L_[...] = np.sum([desc.A_ for desc in descendents], axis=0)
        self.T_[...] = self.model.transition_matrix(ancestor_branch_length)
        self.A_[...] = log_safe_matmul(self.T_, self.L_) #(N_states, N_sites)



class RootNode(Node):

    def __init__(self,
                 name : str,
                 model,
                 initial_state : np.ndarray,
                ) -> None:
        
        assert initial_state.shape == (model.n_states,)

        self.name = name
        self.model = model
        self.initial_state = initial_state

        self.L_ = np.zeros((model.n_states, model.site_dim))
        self.B_ = np.nan_to_num(np.log(self.initial_state[:, np.newaxis]), nan=-np.inf)


    def forward(self,tree):
        pass

    def backward(self, tree):
        descendents = get_descendents(tree, self)
        self.L_[...] = np.sum([desc.A_ for desc in descendents], axis=0) # (N_states, N_sites)
        self.ll_per_site_ = log_safe_vecmul(self.initial_state, self.L_).ravel()*self.model.site_weight_vector

    @property
    def ll_per_site(self):
        return self.ll_per_site_
    


class BranchlenOptimizer:

    @staticmethod
    def _interleaf_distance_obj_grad(nodeI, nodeJ, nu):
        
        def fn(mat):
            return ( np.exp(nodeI.L_) * (mat(nu) @ np.exp(nodeJ.L_)) ).sum(axis=0)
            #return np.exp( nodeI.L_ + log_safe_matmul(mat(nu), nodeJ.L_) ).sum(axis=0)
        
        f=fn(nodeI.model.transition_matrix)
        f_prime=fn(nodeI.model.ddt_transition_matrix)
        f_prime_prime=fn(nodeI.model.dddt_transition_matrix)

        weights = nodeI.model.site_weight_vector
        obj = np.sum(weights*np.log( f ))
        grad = np.sum( weights*f_prime / f )
        hess = np.sum( weights*f_prime_prime/f - (f_prime/f)**2 )

        return -obj, -grad, -hess

    @classmethod
    def estimate_internode_distance(cls, nodeI, nodeJ, nu=0.1, min_bl=1e-6, max_bl=100, tol=1e-6):
    
        last_obj=-np.inf
        for i in range(100):
            
            obj, grad, inv_score = cls._interleaf_distance_obj_grad(nodeI, nodeJ, nu)
            
            if i > 0 and (min_bl < nu < max_bl) and  ( (obj > last_obj + tol) or (np.abs(inv_score) < tol) ):
                nu /= 10
                #logger.debug(f'Newton optimization stuck on a plateau, jumping to new spot: {nu:.3f}')
            #elif (inv_score > 1e4):
            #    nu *= 10
                #logger.debug(f'Newton optimization stuck on a plateau, jumping to new spot: {nu:.3f}')
            else:
                nu -= grad/inv_score
                nu=np.clip(nu, min_bl, max_bl)

            last_obj=obj
            if np.abs(grad) < tol:
                break
        else:
            raise ValueError('Newton optimization did not converge')

        return nu, 1/inv_score

        '''res = minimize(
            lambda x : cls._interleaf_distance_obj_grad(nodeI, nodeJ, x),
            nu,
            jac=True,
            bounds=Bounds(min_bl, max_bl),
            tol=tol,
            method='L-BFGS-B'
        )
        assert res.success

        return res.x[0], 0'''


    @staticmethod
    def forward_backward_pass(tree):

        root_node=get_root(tree)
        dfs_nodeset = get_nodes_dfs(tree, root_node)

        for node in dfs_nodeset:
            node.backward(tree)

        for node in reversed(dfs_nodeset):
            node.forward(tree)
    

    @classmethod
    def get_tree_ll(cls,tree):
        cls.forward_backward_pass(tree)
        root_node=get_root(tree)
        return root_node.ll_per_site.sum()
    
    
    @classmethod
    def get_tree_ll_from_node(cls, node):
        pass    
    

    @staticmethod
    def get_path_matrix(tree, root_node=None):

        root_node=get_root(tree) if root_node is None else root_node
        edge_list=get_edges_dfs(tree, root_node)
        leaves_list=get_leaves(tree)
        n_leaves = len(leaves_list)
        path_matrix = np.zeros((len(edge_list), n_leaves), dtype=int)

        active_edges=set()

        def dfs(node):
            if tree.out_degree(node) == 0:
                for active_edge in active_edges:
                    path_matrix[edge_list.index(active_edge), leaves_list.index(node)] = 1
            else:
                for child in tree.successors(node):
                    active_edges.add((node, child))
                    dfs(child)
                    active_edges.remove((node, child))

        dfs(root_node)

        return path_matrix.T


    @classmethod
    def ultrametric_projection(cls, tree, tree_len=None, blmin=1e-3, blmax=100):

        first_brach=get_descendents(tree, get_root(tree))[0]
        D = cls.get_path_matrix(tree, first_brach)
        nu = get_branchlen_vector(tree, first_brach)

        if tree_len is None:
            tree_len = np.median(D @ nu)

        #d = tree_len - D @ nu
        #nu_ult = nu + ( D.T @ np.linalg.inv(D @ D.T) ) @ d
        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            res = minimize(
                lambda x : 0.5*np.square(x).sum(),
                nu,
                jac = lambda x : x,
                bounds=Bounds(-nu + 1e-3, blmax - nu),
                tol=1e-6,
                method='SLSQP',
                constraints=[
                    {'type':'eq', 
                    'fun': lambda x : D @ x + D @ nu - tree_len,
                    'jac': lambda x : D
                    }
                ]
            )
        assert res.success, res.message + '\n' + str(res)
        nu_ult = nu + res.x

        for i, edge in enumerate(get_edges_dfs(tree, first_brach)):
                set_branchlen(tree, edge, np.clip(nu_ult[i], blmin, blmax))

        return tree


    @staticmethod
    def obj_grad_branchlen(nu,*,tree, edge):
        
        nodeI, nodeJ = edge
        try:
            A = get_sibling(tree, nodeJ).A_
        except NoSiblingError:
            A = 0.
        
        log_state_weights = nodeI.B_ + A
        weight = nodeI.model.site_weight_vector

        f = np.exp(nodeJ.A_ + log_state_weights).sum(0)
        f_prime = ((nodeI.model.ddt_transition_matrix(nu) @ np.exp(nodeJ.L_)) * np.exp(log_state_weights)).sum(0)

        with np.errstate(divide='ignore', invalid='ignore'):
            obj = np.inner(weight, np.log(f))
            grad = np.inner(weight, np.where(f_prime==0, 0., f_prime/f))

        return obj, grad

    
    @classmethod
    def optimize_branchlens(cls, tree, constrain=False, **kw):

        root_node=get_root(tree)

        edgelist=get_edges_dfs(tree, root_node)
        dfs_nodelist=get_nodes_dfs(tree, root_node)

        return cls._optimize(tree, edgelist, dfs_nodelist, constrain=constrain,**kw)


    @classmethod
    def _optimize(cls, tree : nx.DiGraph, 
                 edgelist, 
                 dfs_nodelist, 
                 blmin=0.01, 
                 blmax=100,
                 constrain=False,
                ):

        def obj_jac(nu):
            
            for edge, _nu in zip(edgelist, nu):
                set_branchlen(tree, edge, _nu)

            for node in dfs_nodelist:
                node.backward(tree)

            for node in reversed(dfs_nodelist):
                node.forward(tree)

            jac=np.zeros(len(nu))
            for i, edge in enumerate(edgelist):
                obj, grad = cls.obj_grad_branchlen(get_branchlen(tree, edge), tree=tree, edge=edge)
                jac[i] = grad

            return -obj, -jac
        
        init_nu=np.array([get_branchlen(tree, edge) for edge in edgelist])

        if not constrain:
            
            res = minimize(
                obj_jac,
                init_nu,
                jac=True,
                bounds=Bounds(blmin, blmax),
                tol=1e-6,
                method='L-BFGS-B'
            )
        else:
            
            #BranchlenOptimizer.ultrametric_projection(tree, blmin=blmin, blmax=blmax)
            
            path_matrix = cls.get_path_matrix(tree)
            path_lengths = path_matrix @ init_nu
            treelen = path_lengths.mean()

            constraints=[
                {'type':'eq', 
                 'fun': lambda x: path_matrix @ x - treelen,
                 'jac': lambda x: path_matrix
                }
            ]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                res = minimize(
                    obj_jac,
                    init_nu,
                    jac=True,
                    method='SLSQP',
                    bounds=Bounds(blmin, blmax),
                    options={'disp':False, 'maxiter': 100},
                    constraints=constraints,
                    tol=1e-6,
                    callback=lambda x: None
                )

        if not res.success:
            logger.debug('Branch length optimization did not converge.')
            #raise ValueError('Branch length optimization did not converge.\n' + str(res))
        
        #for edge, nu in zip(edgelist, res.x):
        #    set_branchlen(tree, edge, nu)

        return tree, -res.fun


    @staticmethod
    def _tree_depth_obj_grad(tree, nu):

        initial_state = np.exp(tree.graph['root'].initial_state)[:, np.newaxis]
        leaf_nodes = get_leaves(tree)
        model = leaf_nodes[0].model
        
        def fn(mat):
            return np.array([
                ( initial_state * (mat(nu) @ np.exp(node.L_)) ).sum(axis=0)
                for node in leaf_nodes
            ]).sum(0)
        
        f=fn(model.transition_matrix)
        f_prime=fn(model.ddt_transition_matrix)
        f_prime_prime=fn(model.dddt_transition_matrix)

        weights = model.site_weight_vector
        obj = np.sum(weights*np.log( f ))
        grad = np.sum( weights*f_prime / f )
        hess = np.sum( weights*f_prime_prime/f - (f_prime/f)**2 )

        return -obj, -grad, -hess
    

    @staticmethod
    def objective_trace(fn, min=0.01, max=100, steps=100):
        x=np.linspace(min, max, steps)
        y=np.array([fn(xx)[0] for xx in x])
        return {'x':x, 'y':y}
        