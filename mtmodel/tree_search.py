from .tree import *
from .node import BranchlenOptimizer
#from joblib import Parallel, delayed
import numpy as np
import tqdm
import logging
logger = logging.getLogger('HillClimbNNI')


class NoNNIsAppliedError(Exception):
    pass

class TreeSearchNNI:

    @classmethod
    def stochastic_NNI(cls, tree, perturbation=0.5):
        pass


    @classmethod
    def rank_NNIs(cls, tree, quiet=False, **kw):
        
        root_node=get_root(tree)
        current_ll = BranchlenOptimizer.get_tree_ll(tree)

        def nni_generator():
            edge_it=get_edges_dfs(tree, root_node)
            if not quiet:
                edge_it=tqdm.tqdm(edge_it, total=tree.number_of_edges(), desc='NNI search')

            for edge in edge_it:
                parent, child = edge
                if tree.out_degree(child) == tree.out_degree(parent) == 2:
                    for switch in [1,2]:
                        yield (edge, switch)

        results={}
        for (edge, switch) in nni_generator():
            
            edge_index=list(tree.edges).index(edge)
            nni_tree=deepcopy(tree)
            edge_copy=list(nni_tree.edges)[edge_index]

            nni_fn = get_NNI_1 if switch==1 else get_NNI_2
            nni_tree, optim_edges, optim_nodes, blacklist_edges = nni_fn(nni_tree, edge_copy)

            update_node_suffstats = list(optim_nodes) \
                + get_nodes_dfs(nx.reverse(nni_tree, copy=False), edge_copy[0])[::-1]
            
            for node in update_node_suffstats:
                node.backward(nni_tree)

            for node in reversed(update_node_suffstats):
                node.forward(nni_tree)

            nni_tree, nni_ll = BranchlenOptimizer._optimize(
                nni_tree, 
                edgelist=optim_edges,
                dfs_nodelist=optim_nodes,
                **kw,
            )

            edgeset_key = set(((p.name, c.name) for p,c in blacklist_edges))
            results[(edge, switch)] = (nni_ll - current_ll, edgeset_key)

        return sorted(results.items(), key=lambda x: x[1][0], reverse=True)


    @classmethod
    def apply_NNIs(cls, tree, nni_searchresults, iter, **kw):
        
        #current_ll = BranchlenOptimizer.get_tree_ll(tree)
        tau=1.
        
        modified_edges = set()
        nnis_applied = 0
        for (edge_key, switch), (delta_ll, blacklist_edges) in nni_searchresults:
            
            if np.exp(delta_ll*iter*tau) > np.random.rand() and len( modified_edges.intersection(blacklist_edges) )==0:
                
                if delta_ll < 0:
                    logger.debug(f' Applying NNI with negative log-likelihood change: {delta_ll:.3f}')

                try:
                    nni_fn = get_NNI_1 if switch==1 else get_NNI_2
                    nni_fn(tree, edge_key)
                    
                    #for edge, new_bl in zip(optim_edges, new_branchlens):
                    #    tree.edges[edge]['branch_length'] = new_bl
                    
                    modified_edges.update(blacklist_edges)
                    nnis_applied += 1
                except StopIteration:
                    logger.warn('Incompatible NNI')
                except KeyError:
                    logger.warn('Edge not found')
                    
            
        if nnis_applied==0:
            raise NoNNIsAppliedError('No NNIs applied')
        
        logger.info(f' Applied {nnis_applied} NNIs')
        
        return tree
    
    
    @classmethod
    def hill_climb_NNI(cls, tree, niters=10, quiet=False,**kw):
        
        ll=BranchlenOptimizer.get_tree_ll(tree)
        obj=[]
        logger.info(f' Initial log-likelihood: {ll:.3f}')

        tree, _ =BranchlenOptimizer.optimize_branchlens(tree, **kw)
        ll = BranchlenOptimizer.get_tree_ll(tree)
        #obj.append(ll)
        #logger.info(f' Iteration 1, Log-likelihood: {ll:.3f}')

        for i in range(1, niters):
            
            #BranchlenOptimizer.ultrametric_projection(tree, **kw)
            nni_searchresults = cls.rank_NNIs(tree, quiet=quiet, **kw)
            
            try:
                tree = cls.apply_NNIs(tree, nni_searchresults, i, **kw)
            except NoNNIsAppliedError as e:
                break
            else:
                tree, _ = BranchlenOptimizer.optimize_branchlens(tree, **kw)
                ll = BranchlenOptimizer.get_tree_ll(tree)
                logger.info(f' Iteration {i}, Log-likelihood: {ll:.3f}')
                obj.append(ll)

            with open(f'testdata/hillclimb.2.{i}.nwk', 'w') as f:
                f.write(to_newick(tree))
        
        logger.debug(f' Hill climb completed.')

        return tree, obj