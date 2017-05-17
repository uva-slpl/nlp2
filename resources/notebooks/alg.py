"""
Here I implement some graph algorithms.

:author: Wilker Aziz
"""

from collections import defaultdict
from formal import Symbol, CFG, FSA


def iter_useful_edges(forest: CFG, root: Symbol) -> CFG:
    """
    This algorithm performs a recursive cleanup of the forest by deleting edges which will surely have 0 probability
    without actually computing probabilities.
    
    We do so by running inside-outside algorithm in the boolean semiring, 
    this tells us whether nodes are relevant to the forest (because it tells us whether some complete derivation
    goes through that node). 

    This is a **recursive** that does not require top-sorting and works even with cyclic forests.
    The reason why this works with cyclic forests is that in the boolean semiring there's a neat solution to avoid infinite loops
    (check below).

    :param forest: a cyclic or acyclic forest
    :param root: Nonterminal/Span that represents the root of the forest
    :returns: a generator of useful edges

    :author: Wilker Aziz
    """
    value = defaultdict(int)  # maps from nodes to boolean value
    fwd_star = defaultdict(set)
    rev_value = defaultdict(int)
        
    for edge in forest:
        for child in edge.rhs:
            fwd_star[child].add(edge)
    
    def get_boolean_value(node):  
        # first we check whether we have a color for this node
        color = value.get(node, None)        
        if color is None:  # here we don't
            value[node] = -1  # but we mark the node so we know are visiting it
        elif color == -1:  # we are visiting this node
            return True  # so we just assume things went well
        else:  # here we have color 1 (which implies True) or color 0 (which implies False)
            return color == 1
        
        # here we need to compute the node's color
        bs = forest.get(node)
        if not bs:  # no incoming edges
            flag = node.is_terminal()  # then color depends on type of symbol, if Terminal, color is 1 (as in True)
            color = 1 if node.is_terminal() else 0
        else:
            # there are incoming edges, we check whether there are valid paths below the node
            n = 0
            for edge in bs:  # for each incoming edge
                for child in edge.rhs:  # for each child node
                    if get_boolean_value(child):  # flag = flag or get_boolean_value(child)
                        n += 1
            color = 1 if n > 0 else 0
        value[node] = color
        return color == 1

    def get_boolean_rev_value(node):
        color = rev_value.get(node, None)
        if color is None:  # here we don't
            rev_value[node] = -1  # but we mark the node so we know are visiting it
        elif color == -1:  # we are visiting this node
            return True  # so we just assume things went well
        else:  # here we have color 1 (which implies True) or color 0 (which implies False)
            return color == 1

        fs = fwd_star.get(node, {})
        
        if not fs:  # no outgoing edge
            color = 1 if node == root else 0  # this must be the root to get 1
        else:
            n = 0
            for edge in fs:
                edge_rev = get_boolean_rev_value(edge.lhs)
                edge_value = True
                for sibling in edge.rhs:
                    if sibling != node:
                        if value.get(sibling, 0) != 1:
                            edge_value = False
                            break
                if edge_rev and edge_value:
                    n += 1
            color = 1 if n > 0 else 0
        rev_value[node] = color
        return color == 1

    get_boolean_value(root)
    for node, color in value.items():
        if node.is_terminal() and color == 1 and node not in rev_value:
            get_boolean_rev_value(node)

    for edge in forest:
        if value[edge.lhs] != 1 or rev_value[edge.lhs] != 1:
            continue
        selected = True
        for child in edge.rhs:
            if value[child] != 1 or rev_value[child] != 1:
                selected = False
                break
        if selected:
            yield edge


def cleanup_forest(forest: CFG, root: Symbol) -> CFG:
    """This wraps iter_useful_edges and return a clean CFG where every edge is useful"""
    return CFG(iter_useful_edges(forest, root))


# # Enumerating paths

def forest_to_fsa(forest: CFG, start_symbol: Symbol) -> FSA:
    """
    Note that this algorithm only works with acyclic forests.
    Even for such forests, this runs in exponential time, so make sure to only try it with very small forests.
    
    :param forest: acyclic forest
    :param start_symbol:
    :return FSA
    """    
    fsa = FSA()

    # here we find out which spans end in an accepting state (the spans of top rules contain that information)
    #accepting = set()
    #for rule in forest.iter_rules(start_symbol):  # S' -> S:initial-final
    #    for sym in rule.rhs:  # the RHS contains of top rules contain the accepting states 
    #        s, initial, final = sym.obj()
    #        accepting.add(final)
   
    def visit_forest_node(symbol: Symbol, bos, eos, parent: Symbol):
        """Visit a symbol spanning from bos to eos given a parent symbol"""
        if symbol.is_terminal():
            fsa.add_arc(bos, eos, symbol.root().obj())
            #if isinstance(parent, Span) and parent.obj()[-1] in accepting:
            #    fsa.make_final(eos)
        else:
            for rule in forest.get(symbol):
                # generate the internal states
                states = [bos]
                states.extend([fsa.add_state() for _ in range(rule.arity - 1)])
                states.append(eos)            
                # recursively call on nonterminal children
                for i, child in enumerate(rule.rhs):
                    visit_forest_node(child, states[i], states[i + 1], symbol)
    
    fsa.add_state(initial=True)  # state 0
    fsa.add_state(final=True)  # state 1
    visit_forest_node(start_symbol, 0, 1, None)

    return fsa


def language_of_fsa(fsa: FSA, eps_str='-EPS-') -> set:
    """Return the set of strings in the FSA: this runs in exponential time, use with very small FSA only"""
    # then we enumerate paths in this FSA
    #from collections import Counter
    strings = set()
    
    def visit_fsa_state(state, string: tuple):
        if fsa.is_final(state):
            strings.add(' '.join(x for x in string))  #
        for label, destinations in fsa.iterarcs(state, group_by='label'):
            if label != eps_str:
                for destination in destinations:
                    visit_fsa_state(destination, string + (label,))
            else:
                for destination in destinations:
                    visit_fsa_state(destination, string)
   
    for initial in fsa.iterinitial():
        visit_fsa_state(initial, tuple())
   
    return strings


