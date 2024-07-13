import jax.numpy as jnp
import jax
from jax import jit, vmap
from functools import partial
from .binarytree import BinaryTree


@jit
def logsafe_matmul(A, logB):
    eps = jnp.max(logB)
    logmat = jnp.nan_to_num(jnp.log(A @ jnp.exp(logB - eps)) + eps, nan=-jnp.inf)
    return logmat


@jit
def exp_logsafe_matmul(A, logB):
    eps = jnp.max(logB)
    return A @ jnp.exp(logB - eps) * jnp.exp(eps)


@jit
def logsumexp_down(A):
    eps = jnp.max(A)
    return jnp.log(jnp.sum(jnp.exp(A - eps), axis=0)) + eps


@jit
def transition_matrix(
    t,
    *,
    U,
    inv_U,
    D,
):
    """
    Parameters
    ----------
    t : float
        Branch length
    U : ndarray (N, N)
        Eigenvectors
    inv_U : ndarray (N, N)
        Inverse of eigenvectors
    D : ndarray (N,1)
        Diagonalization vector
    """
    return (U * jnp.exp(D * t)) @ inv_U


@jit
def dT_transition_matrix(Q, transition_matrix):
    """
    Parameters
    ----------
    transition_matrix : ndarray (N, N)
        Transition matrix
    Q : ndarray (N, N)
        Derivative of the transition matrix
    """
    return transition_matrix @ Q
    

@jit
def unstack_tensor(tensor):
    return jax.tree.map(jnp.squeeze, jnp.split(tensor, tensor.shape[0]))


def parallelize_transition_fn(transition_matrix_fn, **kw):
    map_fn = partial(transition_matrix_fn, **kw)
    return jit(vmap(map_fn))


@partial(jit, static_argnums=(0,1,2,))
def _init_tensors(
        n_nodes,
        n_states,
        n_sites,*,
        leaf_node_idxs,
        logp_initial_state,
        root_idx,
        logp_obs_arr,
        T,
    ):
    '''
    Initialize the tensors for the message passing algorithm with zeros,
    then filling in the known sections. The transition matrices at each node
    can be calculated in parallel, as can the backward messages at the leaves.

    In the signature, transition_matrix_fn, n_nodes, n_states, and n_sites
    are static arguments against which the function is compiled. Since these
    values won't change during the computation, we can target the function
    to these specific values.
    '''

    # set up tensors to store the forward and backward messages, initialized to 0
    L_rev = jnp.empty((n_nodes, n_states, n_sites))\
            .at[root_idx, :, :].set(logp_initial_state)

    B = jnp.empty((n_nodes, n_states, n_sites))\
            .at[root_idx, :, :].set(
                logsafe_matmul(T[root_idx].T, L_rev[root_idx])
            )
    
    # initialize L matrices for leaf nodes
    L = jnp.empty((n_nodes, n_states, n_sites))\
            .at[leaf_node_idxs, :, :].set(
                logp_obs_arr
            )
    
    # parallelize the first backward step for all leave nodes
    A = jnp.empty((n_nodes, n_states, n_sites))\
            .at[leaf_node_idxs, :, :].set(
                logsafe_matmul(T[leaf_node_idxs, :, :], L[leaf_node_idxs, : , :])
            )
    
    return L_rev, L, A, B


@jit
def backward_pass(curr_idx, left_idx, right_idx,*, T, A, L):
    L=L.at[curr_idx, :, :].set(A[left_idx, :, :] + A[right_idx,:,:])
    A=A.at[curr_idx, :, :].set(logsafe_matmul(T[curr_idx,:,:], L[curr_idx,:,:]))
    return L, A


@jit
def forward_pass(curr_idx, ancestor_idx, sibling_idx,*, T, B, A, L_rev):
    L_rev=L_rev.at[curr_idx, :, :].set(B[ancestor_idx, :, :] + A[sibling_idx, :, :])
    B=B.at[curr_idx, :, :].set(logsafe_matmul( jnp.matrix_transpose(T[curr_idx, :, :]), L_rev[curr_idx, :, :]))
    return B, L_rev


def get_gradient_suffstat_fn(
    transition_matrix_fn,
    tree,
    *,
    logp_obs_dict,
    logp_initial_state,
):
    '''
    For a fixed topology, we can precompute and fix the forward and backward message 
    paths. This saves time in the optimization loop, as we can repeatedly call the 
    gradient suffstat function with different branch lengths while keeping the
    message paths fixed.
    
    TODO:
    - If the branchlens of a subtree are not updated, we can re-use the
        backward messages from the previous run.
    - The transition matrix can be cached for each branchlen value.
    - We can use a taylor's heap for the transition matrix to speed up its calculation.
    - The branchlen information should not be attached to the tree, but rather
        passed around as an array.
    '''

    nodes = BinaryTree.list_nodes(tree)
    n_nodes=len(nodes)
    n_states, n_sites = logp_initial_state.shape
    tensor_shape = (n_nodes, n_states, n_sites)
    
    idx_map = {node.id : i for i, node in enumerate(nodes)}
    root_idx=idx_map[tree.id]
    leaf_nodes=[node for node in nodes if node.is_leaf]
    leaf_nodes_idxs = [idx_map[node.id] for node in leaf_nodes]

    logp_obs_arr=jnp.array([logp_obs_dict[node.id] for node in leaf_nodes])

    # Sort the nodes by level - e.g. how far they are from the leaves.
    # Since in the backward pass no node depends on information from 
    # any node at or above its level, we can parallelize the computation.
    levels = sorted(
        BinaryTree.list_distances_from_leaves(tree).items(), key=lambda x: x[0]
    )

    # don't do the leaves - these are special cases which we handle separately
    # in the _init_tensors function.
    backward_nodes=[]
    for _, levelnodes in levels[1:]:
        backward_nodes.append(
            list(zip(*(
                (idx_map[node.id], idx_map[node.left.id], idx_map[node.right.id]) 
                for node in levelnodes
            )))
        )

    forward_nodes=[]
    for _, levelnodes in levels[:-1][::-1]:
        forward_nodes.append(
            list(zip(*(
                (idx_map[node.id], idx_map[node.ancestor.id], idx_map[node.sibling.id]) 
                for node in levelnodes
            )))
        )


    def get_grad_suffstats(branchlen_vector):

        # calculate the transition matrix for each branch
        # this is separated because it will be computed using a 
        # TaylorHeap in the future.
        T = transition_matrix_fn(branchlen_vector)
    
        L_rev, L, A, B = _init_tensors(
            *tensor_shape,
            logp_initial_state=logp_initial_state,
            root_idx=root_idx,
            leaf_node_idxs=tuple(leaf_nodes_idxs),
            logp_obs_arr=logp_obs_arr,
            T=T,
        )

        # iterate over the levels, starting from the leaves' parent nodes
        for curr_idxs, left_idxs, right_idxs in backward_nodes:
            L, A = backward_pass(
                        curr_idxs, left_idxs, right_idxs, 
                        T=T, A=A, L=L
                    )

        # forward message at the root node is already calculated
        # iterate over the levels in reverse order
        for curr_idxs, ancestor_idxs, sibling_idxs in forward_nodes:
            B, L_rev = forward_pass(
                            curr_idxs, ancestor_idxs, sibling_idxs, 
                            T=T, A=A, B=B, L_rev=L_rev
                        )

        logf = logsumexp_down(B[root_idx] + L[root_idx])

        return T, L_rev, L, logf
    
    return get_grad_suffstats



def get_gradient_suffstats_slow_but_correct(
    transition_matrix_fn,
    ordered_nodes,  # the nodes to update should be in DFS preorder order
    *,
    logp_obs_dict,
    logp_initial_state,
    precomputed_L_unstack={},  # one can initialize the gradient suffstats using a previous run's `L_unstack`, for example, if only updating the branch lengths on a subset of nodes.
):
    """
    Update the gradient sufficient statistics for a given tree.

    """

    idxs = [node.id for node in ordered_nodes]
    branchlen_vector = jnp.array([node.branchlen for node in ordered_nodes])
    n_sites = next(iter(logp_obs_dict.values())).shape[1]

    # batched computation of the transition matrix
    T_unstack = dict(zip(idxs, unstack_tensor(transition_matrix_fn(branchlen_vector))))

    # set up dictionaries to store the forward and backward messages, initialized to None
    # so that the keys are in the same order
    B_unstack = {k: None for k in idxs}
    A_unstack = {k: None for k in idxs}
    L_unstack = {k: None for k in idxs}
    L_rev_unstack = {k: None for k in idxs}

    for node in ordered_nodes:
        if node.is_leaf:
            # for leaf nodes, L_ is pre-computed, use this always
            L_node = jnp.array(logp_obs_dict[node.id])
        elif node.id in precomputed_L_unstack and not (
            node.left.id in A_unstack and node.right.id in A_unstack
        ):
            # if the node is not a leaf, the L_ matrix is pre-computed, and its children are not in A_unstack
            L_node = precomputed_L_unstack[node.id]
        else:
            left_child, right_child = node.children
            L_node = A_unstack[left_child.id] + A_unstack[right_child.id]

        L_unstack[node.id] = L_node
        A_unstack[node.id] = logsafe_matmul(T_unstack[node.id], L_node)

    for node in reversed(ordered_nodes):
        if node.is_root:
            L_rev_unstack[node.id] = jnp.array(logp_initial_state) * jnp.ones(
                (1, n_sites)
            )
        else:
            ancestor = node.ancestor
            sibling = node.sibling
            L_rev_unstack[node.id] = B_unstack[ancestor.id] + A_unstack[sibling.id]

        if not node.is_leaf:
            B_unstack[node.id] = logsafe_matmul(
                T_unstack[node.id].T, L_rev_unstack[node.id]
            )
        # update L_rev

    use_node = ordered_nodes[
        -1
    ]  # compute the logl of the tree using the most senior node
    logf = logsumexp_down(B_unstack[use_node.id] + L_unstack[use_node.id])

    return {
        "L_rev_unstack": L_rev_unstack,
        "L_unstack": L_unstack,
        "logf": logf,
        "A_unstack": A_unstack,
        "T_unstack": T_unstack,
    }


@partial(jit, static_argnums=(0,))
def compute_grad_single(
    dT_transition_matrix_fn,
    T, 
    L_rev, 
    L,
    *,
    site_weights,
    logf
):
    w = L_rev - logf
    eps = jnp.max(w)
    return jnp.inner(
        site_weights,
        jnp.sum(jnp.exp(w - eps) * exp_logsafe_matmul(dT_transition_matrix_fn(T), L), axis=0)
        * jnp.exp(eps),
    )


@partial(jit, static_argnums=(0,))
def compute_f_grad(
    dT_transition_matrix_fn,
    T,
    L_rev,
    L, 
    logf,
    *,
    site_weights,
):
    """
    A simple jitted parallelized function to compute the objective and gradient
    for a while tree in parallel.
    """
    obj = jnp.inner(site_weights, logf)

    grad = vmap(
        partial(
            compute_grad_single,
            dT_transition_matrix_fn,
            logf=logf,
            site_weights=site_weights,
        ),
    )(T, L_rev, L)

    return obj, grad


@partial(jit, static_argnums=(0, 1))
def nearest_neighbor_interchange_grad_obj(
    transition_matrix_fn,
    dT_transition_matrix_fn,
    branchlen_vector,  # ordered l, m, j, k, i
    *,
    site_weights,
    L_rev_i, # = B_g + A_h
    L_k,
    L_l,
    L_m,
):
    """
    Uses the following layout:

       H 
      / 
     G    K
      \\ //
        I    M
        \\ //
          J*
           \\
             L
        
    Where J* is the node on which the NNI is performed, 
    and the double lines are the optimized edges.

    This is a hand-written version of the algorithm used in "get_gradient_suffstats",
    since the latter is a more general version that can be used for any tree.
    This enables jit compilation of the more specialized NNI update routine
    which produces major speedups.
    """

    T_l = transition_matrix_fn(branchlen_vector[0])
    T_m = transition_matrix_fn(branchlen_vector[1])
    T_j = transition_matrix_fn(branchlen_vector[2])
    T_k = transition_matrix_fn(branchlen_vector[3])
    T_i = transition_matrix_fn(branchlen_vector[4])

    # backward pass
    A_l = logsafe_matmul(T_l, L_l)
    A_m = logsafe_matmul(T_m, L_m)
    A_k = logsafe_matmul(T_k, L_k)
    L_j = A_l + A_m
    A_j = logsafe_matmul(T_j, L_j)
    L_i = A_j + A_k
    A_i = logsafe_matmul(T_i, L_i)

    # foward pass
    B_i = logsafe_matmul(T_i.T, L_rev_i)
    L_rev_k = B_i + A_j
    L_rev_j = B_i + A_k
    B_j = logsafe_matmul(T_j.T, L_rev_j)
    L_rev_l = B_j + A_m
    L_rev_m = B_j + A_l

    logf = logsumexp_down(L_rev_i + A_i)
    obj = jnp.inner(site_weights, logf)

    grad_fn = partial(
        compute_grad_single,
        dT_transition_matrix_fn,
        site_weights=site_weights,
        logf=logf,
    )

    grad = jnp.array([
            grad_fn(T_l, L_rev_l, L_l),
            grad_fn(T_m, L_rev_m, L_m),
            grad_fn(T_j, L_rev_j, L_j),
            grad_fn(T_k, L_rev_k, L_k),
            grad_fn(T_i, L_rev_i, L_i),
        ])

    return -obj, -grad


"""
--------------------------------
Computation of sufficient statistics for the NNI optimization.
Particularly, one must compute a new "L_rev_i" for each NNI,
which can be boiled down to a backward-foward pass on the tree spine.
--------------------------------
"""
@jit
def _traverse_spine_step(i, val, *, T, A):
    """
    One step in the reverse spine traversal
    """
    return logsafe_matmul(T[i], val + A[i])


@jit
def _traverse_spine(T_spine, A_rib, A_leaf):
    """
    In the following spine:
    0 --- 2 --- 5
    |     |
    1     6

    The "A" matrix is the following:
    [ T @ L_6, T @ L_1 ]

    The "T" matrix is the following:
    [ T_5, T_2 ]

    The initial value is L_5, and the result is L_0
    """
    T_spine = jnp.array(T_spine)
    A_rib = jnp.array(A_rib)

    return jax.lax.fori_loop(
        0,
        len(T_spine),
        partial(_traverse_spine_step, T=T_spine, A=A_rib),
        A_leaf,
    )


def traverse_spine(
    *,
    leaf_node,
    A_unstack,
    T_unstack,
):

    spine_nodes = list(leaf_node.trace_ancestors())
    ribs = list(leaf_node.ribs())

    A_rib = [A_unstack[node.id] for node in ribs]
    T_spine = [T_unstack[node.id] for node in spine_nodes]
    A_leaf = A_unstack[leaf_node.id]

    return _traverse_spine(T_spine, A_rib, A_leaf)


@jit
def _traverse_spine_rev_step(i, B_anc, *, T_spine, A_sib):
    """
    One step in the reverse spine traversal
    """
    return logsafe_matmul(T_spine[i].T, B_anc + A_sib[i])


@jit
def _traverse_spine_rev(
    T_spine,
    A_rib,
    L_rev_root,
):
    """
    In the following spine:
    0 --- 2 --- 5
    |     |
    1     6

    The "A" matrix is the following:
    [ T @ L_6, T @ L_1 ]

    The "T" matrix is the following:
    [ T_5, T_2 ]

    The initial value is L_5, and the result is L_0
    """
    T_spine = jnp.array(T_spine)
    A_rib = jnp.array(A_rib)

    return jax.lax.fori_loop(
        0,
        len(T_spine),
        partial(_traverse_spine_rev_step, T=T_spine, A_sib=A_rib),
        L_rev_root,
    )


def traverse_spine_rev(
    *,
    leaf_node,
    A_unstack,
    T_unstack,
    L_rev_unstack,
):
    """
    In the following spine:

    --- 0 --- 2 --- 5 --- 14*
        |     |     |
        1     6     13

    B_0 = T_0.T @ L^rev_0
    B_2 = T_2.T @ (B_0 + A_1)
    B_5 = T_5.T @ (B_2 + A_6)
    """
    spine_nodes = list(leaf_node.trace_ancestors())[::-1]
    ribs = list(leaf_node.ribs())[::-1]
    L_rev_root = L_rev_unstack[leaf_node.jump_to_root().id]

    A_rib = [A_unstack[node.id] for node in ribs]
    T_spine = [T_unstack[node.id] for node in spine_nodes]

    return _traverse_spine_rev(T_spine, A_rib, L_rev_root)
