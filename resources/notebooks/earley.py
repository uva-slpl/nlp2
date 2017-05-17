"""
Here I implement Earley intersection between an arbitrary CFG (including eps rules) and an epsilon-free NFA.

:author: Wilker Aziz
"""

from collections import defaultdict, deque
from weakref import WeakValueDictionary
from formal import Symbol, Terminal, Span
from formal import Rule, CFG, FSA
from alg import cleanup_forest

# ## Items
# 
# First we represent the items of our deductive system (again immutable objects).


"""
An item in a CKY/Earley program.
"""

class Item:
    """A dotted rule used in CKY/Earley where dots store the intersected FSA states."""

    __repositories = defaultdict(WeakValueDictionary)

    def __new__(cls, rule: Rule, dots: list):
        assert len(dots) > 0, 'I do not accept an empty list of dots'
        dots = tuple(dots)
        repository = Item.__repositories[rule]
        instance = repository.get(dots, None)
        if instance is None:
            instance = object.__new__(Item)
            instance._rule = rule
            instance._dots = dots
            repository[dots] = instance
        return instance
    
    @classmethod
    def nb_instances(cls):
        """Number of instances of Item"""
        return sum(len(repo) for repo in Item.__repositories.values())

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self) 

    def __repr__(self):
        return '{0} ||| {1}'.format(self._rule, self._dots)

    def __str__(self):
        return '{0} ||| {1}'.format(self._rule, self._dots)

    @property
    def lhs(self) -> Symbol:
        return self._rule.lhs

    @property
    def rule(self) -> Rule:
        return self._rule

    @property
    def dot(self) -> int:
        return self._dots[-1]

    @property
    def start(self) -> int:
        return self._dots[0]

    @property
    def next(self) -> Symbol:
        """return the symbol to the right of the dot (or None, if the item is complete)"""
        if self.is_complete():
            return None
        return self._rule.rhs[len(self._dots) - 1]

    def state(self, i) -> int:
        """The state associated with the ith dot"""
        return self._dots[i]

    def advance(self, dot) -> 'Item':
        """return a new item with an extended sequence of dots"""
        return Item(self._rule, self._dots + (dot,))

    def is_complete(self) -> bool:
        """complete items are those whose dot reached the end of the RHS sequence"""
        return len(self._rule.rhs) + 1 == len(self._dots)


# ## Agenda
# 
# Next we define an agenda of active/passive items. 
# Agendas are much like queues, but with some added functionality (see below).


"""
An agenda of active/passive items in CKY/Ealery program.
"""

class Agenda:

    def __init__(self):
        # we are organising active items in a stack (last in first out)
        self._active = deque([])
        # an item should never queue twice, thus we will manage a set of items which we have already seen
        self._seen = set()
        # we organise incomplete items by the symbols they wait for at a certain position
        # that is, if the key is a pair (Y, i)
        # the value is a set of items of the form
        # [X -> alpha * Y beta, [...i]]
        self._incomplete = defaultdict(set)
        # we organise complete items by their LHS symbol spanning from a certain position
        # if the key is a pair (X, i)
        # then the value is a set of items of the form
        # [X -> gamma *, [i ... j]]
        # (lhs, start) => end => set of items
        self._complete = defaultdict(lambda: defaultdict(set))


    def __len__(self):
        """return the number of active items"""
        return len(self._active)

    def push(self, item: Item):
        """push an item into the queue of active items"""
        if item not in self._seen:  # if an item has been seen before, we simply ignore it
            self._active.append(item)
            self._seen.add(item)
            return True
        return False

    def pop(self) -> Item:
        """pop an active item"""
        assert len(self._active) > 0, 'I have no items left.'
        return self._active.pop()

    def make_passive(self, item: Item):
        """Store an item as passive: complete items are part of the chart, incomplete items are waiting for completion."""
        if item.is_complete():  # complete items offer a way to rewrite a certain LHS from a certain position
            self._complete[(item.lhs, item.start)][item.dot].add(item)
        else:  # incomplete items are waiting for the completion of the symbol to the right of the dot
            self._incomplete[(item.next, item.dot)].add(item)

    def waiting(self, symbol: Symbol, dot: int) -> set:
        """return items waiting for `symbol` spanning from `dot`"""
        return self._incomplete.get((symbol, dot), set())

    def complete(self, lhs: Symbol, start: int) -> set:
        """return complete items whose LHS symbol is `lhs` spanning from `start`"""
        return self._complete.get((lhs, start), dict()).items()
    
    def destinations(self, lhs: Symbol, start: int) -> set:
        """return destinations (in the FSA) for `lhs` spanning from `start`"""
        return self._complete.get((lhs, start), dict()).keys()

# # Deductive system
# 
# We implement our parser using a deductive system.
# 


# ## Inference rules
# 
# Now, let's implement an Earley parser. It is based on a set of *axioms* and 3 inference rules (i.e. *predict*, *scan*, and *complete*).
# 
# The strategy we adopt here is to design a function for each inference rule which
# * may consult the agenda, but not alter it
# * infers and returns a list of potential consequents
# 

def axioms(cfg: CFG, fsa: FSA, s: Symbol) -> list:
    """
    Axioms for Earley.

    Inference rule:
        -------------------- (S -> alpha) \in R and q0 \in I
        [S -> * alpha, [q0]] 
        
    R is the rule set of the grammar.
    I is the set of initial states of the automaton.

    :param cfg: a CFG
    :param fsa: an FSA
    :param s: the CFG's start symbol (S)
    :returns: a list of items that are Earley axioms  
    """
    items = []
    for q0 in fsa.iterinitial():
        for rule in cfg.get(s):
            items.append(Item(rule, [q0]))
    return items

def predict(cfg: CFG, item: Item) -> list:
    """
    Prediction for Earley.

    Inference rule:
        [X -> alpha * Y beta, [r, ..., s]]
        --------------------   (Y -> gamma) \in R
        [Y -> * gamma, [s]] 
        
    R is the ruleset of the grammar.

    :param item: an active Item
    :returns: a list of predicted Items or None  
    """
    return [Item(rule, [item.dot]) for rule in cfg.get(item.next)]

def scan(fsa: FSA, item: Item, eps_symbol: Terminal) -> list:
    """
    Scan a terminal (compatible with CKY and Earley).

    Inference rule:

        [X -> alpha * x beta, [q, ..., r]]
        ------------------------------------    where (r, x, s) \in FSA and x != \epsilon
        [X -> alpha x * beta, [q, ..., r, s]]
        
        
    If x == \epsilon, we have a different rule
    
        [X -> alpha * \epsilon beta, [q, ..., r]]
        ---------------------------------------------   
        [X -> alpha \epsilon * beta, [q, ..., r, r]]
    
    that is, the dot moves over the empty string and we loop into the same FSA state (r)

    :param item: an active Item
    :param eps_symbol: a list/tuple of terminals (set to None to disable epsilon rules)
    :returns: scanned items
    """
    assert item.next.is_terminal(), 'Only terminal symbols can be scanned, got %s' % item.next
    if eps_symbol is not None and item.next.root() == eps_symbol:
        return [item.advance(item.dot)]
    else:
        # we call .obj() because labels are strings, not Terminals
        return [item.advance(destination) for destination in fsa.destinations(origin=item.dot, label=item.next.root().obj())]
        
def complete(agenda: Agenda, item: Item):
    """
    Move dot over nonterminals (compatible with CKY and Earley).

    Inference rule:

        [X -> alpha * Y beta, [i ... k]] [Y -> gamma *, [k ... j]]
        ----------------------------------------------------------
                 [X -> alpha Y * beta, [i ... j]]

    :param item: an active Item.
        if `item` is complete, we advance the dot of incomplete passive items to `item.dot`
        otherwise, we check whether we know a set of positions J = {j1, j2, ..., jN} such that we can
        advance this item's dot to.
    :param agenda: an instance of Agenda
    :returns: a list of items
    """
    if item.is_complete():
        # advance the dot for incomplete items waiting for item.lhs spanning from item.start
        return [incomplete.advance(item.dot) for incomplete in agenda.waiting(item.lhs, item.start)]
    else:
        # look for completions of item.next spanning from item.dot
        return [item.advance(destination) for destination in agenda.destinations(item.next, item.dot)]
    
def earley(cfg: CFG, fsa: FSA, start_symbol: Symbol, sprime_symbol=None, eps_symbol=Terminal('-EPS-'), clean=True):
    """
    Earley intersection between a CFG and an FSA.
    
    :param cfg: a grammar or forest
    :param fsa: an acyclic FSA
    :param start_symbol: the grammar/forest start symbol
    :param sprime_symbol: if specified, the resulting forest will have sprime_symbol as its starting symbol
    :param eps_symbol: if not None, the parser will support epsilon rules
    :param clean: if True, returns a forest without dead edges.
    :returns: a CFG object representing the intersection between the cfg and the fsa 
    """
    
    # start an agenda of items
    A = Agenda()
    
    # this is used to avoid a bit of spurious computation
    have_predicted = set()

    # populate the agenda with axioms
    for item in axioms(cfg, fsa, start_symbol):
        A.push(item)
        
    # call inference rules for as long as we have active items in the agenda
    while len(A) > 0:  
        antecedent = A.pop()
        consequents = []
        if antecedent.is_complete():  # dot at the end of rule                    
            # try to complete other items            
            consequents = complete(A, antecedent)
        else:
            if antecedent.next.is_terminal():  # dot before a terminal 
                consequents = scan(fsa, antecedent, eps_symbol=eps_symbol)
            else:  # dot before a nonterminal
                if (antecedent.next, antecedent.dot) not in have_predicted:  # test for spurious computation
                    consequents = predict(cfg, antecedent)  # attempt prediction
                    have_predicted.add((antecedent.next, antecedent.dot))
                else:  # we have already predicted in this context, let's attempt completion
                    consequents = complete(A, antecedent)
        for item in consequents: 
            A.push(item)
        # mark this antecedent as processed
        A.make_passive(antecedent)

    def iter_intersected_rules():
        """
        Here we convert complete items into CFG rules.
        This is a top-down process where we visit complete items at most once.
        """
        
        # in the agenda, items are organised by "context" where a context is a tuple (LHS, start state)
        to_do = deque()  # contexts to be processed
        discovered_set = set()  # contexts discovered
        top_symbols = []  # here we store tuples of the kind (start_symbol, initial state, final state)
        
        # we start with items that rewrite the start_symbol from an initial FSA state
        for q0 in fsa.iterinitial():
            to_do.append((start_symbol, q0))  # let's mark these as discovered
            discovered_set.add((start_symbol, q0))
                        
        # for as long as there are rules to be discovered
        while to_do:
            nonterminal, start = to_do.popleft()                             
            # give every complete item matching the context above a chance to yield a rule
            for end, items in A.complete(nonterminal, start):
                for item in items:
                    # create a new LHS symbol based on intersected states
                    lhs = Span(item.lhs, item.start, item.dot)
                    # if LHS is the start_symbol, then we must respect FSA initial/final states
                    # also, we must remember to add a goal rule for this
                    if item.lhs == start_symbol:
                        if not (fsa.is_initial(start) and fsa.is_final(item.dot)):
                            continue  # we discard this item because S can only span from initial to final in FSA                        
                        else:
                            top_symbols.append(lhs)
                    # create new RHS symbols based on intersected states
                    #  and update discovered set
                    rhs = []
                    for i, sym in enumerate(item.rule.rhs):
                        context = (sym, item.state(i))
                        if not sym.is_terminal() and context not in discovered_set:
                            to_do.append(context)  # book this nonterminal context
                            discovered_set.add(context)  # mark as discovered
                        # create a new RHS symbol based on intersected states
                        rhs.append(Span(sym, item.state(i), item.state(i + 1)))
                    yield Rule(lhs, rhs)
        if sprime_symbol:
            for lhs in top_symbols:
                yield Rule(sprime_symbol, [lhs])
    # return the intersected CFG :)
    out_forest = CFG(iter_intersected_rules())
    if clean:  # possibly cleaning it first
        out_forest = cleanup_forest(out_forest, sprime_symbol)
    return out_forest


