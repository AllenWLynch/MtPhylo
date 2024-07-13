from .weights import unstack_tensor, nearest_neighbor_interchange_grad_obj, \
    compute_f_grad, get_gradient_suffstat_fn
from .binarytree import *
from functools import partial
import jax.numpy as jnp
from scipy.optimize import minimize, Bounds
from numpy import array, inf
import warnings


def generate_nnis(root_node):
    return (
        (node, about) 
        for node in BinaryTree.list_nodes(root_node) 
        if not (node.is_leaf or node.is_root) and not node.sibling.is_outgroup
        for about in ['left', 'right']
    )


def _wraps_jax_objgrad(fn):
    def wrapper(t):
        obj, grad = fn(t)
        return array(obj), array(grad)
    return wrapper


def generate_nni_optimizations(*,
                root_node, 
                transition_matrix,
                dT_transition_matrix,
                gradient_suffstats, 
                site_weights,
            ):
    '''
    Produce a generator of optimization functions for each NNI in the tree.
    '''

    _, L_rev, L, _ = gradient_suffstats

    nodeids=[node.id for node in BinaryTree.list_nodes(root_node)]
    L_rev_unstack = dict(zip(nodeids, unstack_tensor(L_rev)))
    L_unstack = dict(zip(nodeids, unstack_tensor(L)))

    for node, about in generate_nnis(root_node):

        L, M, J, K, I = BinaryTree.get_nni_neighborhoold(node)

        if about=='left':
            l=K.id
            m=M.id
            k=L.id
            t0 = jnp.array([K.branchlen, M.branchlen, J.branchlen, L.branchlen, I.branchlen])
        else:
            l=L.id
            m=K.id
            k=M.id
            t0 = jnp.array([L.branchlen, K.branchlen, J.branchlen, M.branchlen, I.branchlen])

        obj_grad_fn= _wraps_jax_objgrad(partial(
            nearest_neighbor_interchange_grad_obj,
            transition_matrix,
            dT_transition_matrix,
            L_rev_i = L_rev_unstack[I.id],
            L_l = L_unstack[l],
            L_m = L_unstack[m],
            L_k = L_unstack[k],
            site_weights=site_weights,
        ))

        yield (node, about), (obj_grad_fn, t0)
    

def _fast_lbfgsb_minimize(obj_grad_fn, t0, blmin=1e-2, blmax=100, max_iter=100, **kw):
    '''
    Hehe ... right now it's just a wrapper around scipy.optimize.minimize,
    but it will be replaced with a custom implementation.
    '''
    res = minimize( 
            obj_grad_fn, t0, 
            jac=True, 
            method='L-BFGS-B',
            options=dict(
                maxiter=max_iter,
                ftol=1e-6,
                **kw
            ),
            bounds=Bounds(blmin, blmax),
            **kw
        )
    return res.fun, res.x, res.status


def optimize_nni(obj_grad_fn, t0, **kw):
    '''
    Call a stripped-down version of the L-BFGS-B optimizer which is not GIL-bound
    and can be run in parallel on multiple threads without interruptions.
    '''
    new_ll, _, flag = _fast_lbfgsb_minimize(obj_grad_fn, t0, **kw)
    if not flag==0:
        warnings.warn(f"Optimization failed with flag {flag}")
        return -inf
    
    return -new_ll


def rank_nnis(nni_optimization_puzzles, **kw):
    '''
    Rank the NNI optimizations by the likelihood improvement.
    This could be multi-threaded in the future since it's a simple map operation.
    '''
    nni_lls = (
        (nni, optimize_nni(*puzzle, **kw))
        for nni, puzzle in nni_optimization_puzzles
    )

    return sorted(nni_lls, key=lambda x: x[1], reverse=True)


def _optimize_branchlens(*,
        root_node,
        dT_transition_matrix,
        gradient_suffstats_fn,
        site_weights,
        **kw,
    ):
    '''
    For a fixed topology, optimize the branch lengths.
    '''

    nodeorder=BinaryTree.list_nodes(root_node)
    t0 = array([node.branchlen for node in nodeorder])

    def obj_grad_fn(t):
        obj, grad = compute_f_grad(
            dT_transition_matrix,
            *gradient_suffstats_fn(t),
            site_weights=site_weights
        )
        return -array(obj), -array(grad)

    new_ll, new_bl, flag = _fast_lbfgsb_minimize(obj_grad_fn, t0, **kw)
    
    if not flag==0:
        raise ValueError(f"Optimization failed with flag {flag}")

    for node, bl in zip(nodeorder, new_bl):
        node.branchlen = bl

    return -new_ll, new_bl

    
def _apply_nnis(
        nnis_ranked,
        current_ll,
        acceptance_fn : callable = lambda x : x > 0,
        ):
    '''
    THIS FUNCTION HAS SIDE EFFECTS! The nnis in `nnis_ranked` are applied to the tree.
    Apply the NNI that produces the largest likelihood improvement.
    '''
    applied_nnis=0
    modified_nodes=set()
    for nni, nni_ll in nnis_ranked:
        
        (L,M,J,K,I) = BinaryTree.get_nni_neighborhoold(nni[0])
        neighbors=(L,M,J,K)
        
        if acceptance_fn(nni_ll - current_ll) and not any(n in modified_nodes for n in neighbors):
            BinaryTree.nearest_neighbor_interchange(*nni)
            modified_nodes.update(neighbors)
            applied_nnis+=1

    return applied_nnis



def nni_hill_climb_step(
        *,
        root_node,
        transition_matrix,
        transition_matrix_batch,
        dT_transition_matrix,
        logp_obs_dict,
        logp_initial_state,
        site_weights,
        acceptance_fn = lambda x : x > 0,
        blmin=1e-2,
        blmax=100,
        max_iter=100,
        **optim_kw,
    ):
    '''
    THIS FUNCTION HAS SIDE EFFECTS! The nnis in `nnis_ranked` are applied to the tree.
    '''

    pass_to_optimizers=dict(
                blmin=blmin,
                blmax=blmax,
                max_iter=max_iter,
                **optim_kw,
            )
    
    suffstats_fn = get_gradient_suffstat_fn(
                transition_matrix_fn=transition_matrix_batch,
                tree=root_node,
                logp_obs_dict=logp_obs_dict,
                logp_initial_state=logp_initial_state,
            )

    ll, bl = _optimize_branchlens(
                root_node=root_node,
                dT_transition_matrix=dT_transition_matrix,
                site_weights=site_weights,
                gradient_suffstats_fn=suffstats_fn,
                **pass_to_optimizers,
            )

    nni_puzzles=generate_nni_optimizations(
                root_node=root_node,
                transition_matrix=transition_matrix,
                dT_transition_matrix=dT_transition_matrix,
                gradient_suffstats=suffstats_fn(bl),
                site_weights=site_weights,
            )

    nnis_ranked=rank_nnis(
                nni_puzzles,
                **pass_to_optimizers,
            )
    
    applied_nnis=_apply_nnis(
                nnis_ranked,
                current_ll=ll,
                acceptance_fn =acceptance_fn,
            )

    return ll, applied_nnis



def optimize_branchlens(
        *,
        root_node,
        transition_matrix,
        transition_matrix_batch,
        dT_transition_matrix,
        logp_obs_dict,
        logp_initial_state,
        site_weights,
):
    '''
    Convenient function which has the same API as `nni_hill_climb_step`.
    '''
    
    suffstats_fn = get_gradient_suffstat_fn(
        transition_matrix_fn=transition_matrix_batch,
        tree=root_node,
        logp_obs_dict=logp_obs_dict,
        logp_initial_state=logp_initial_state,
    )

    ll, _ = _optimize_branchlens(
        root_node=root_node,
        dT_transition_matrix=dT_transition_matrix,
        site_weights=site_weights,
        gradient_suffstats_fn=suffstats_fn,
    )

    return ll
