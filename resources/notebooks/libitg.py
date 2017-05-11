
# coding: utf-8

# This notebook should help you with project 2, in particular, it implements a basic ITG parser.

# # Symbols
# 
# Let's start by defining the symbols that can be used in our grammars.
# 
# We are going to design symbols as immutable objects. 
# 
# * a symbol is going to be a container
# * Terminal and Nonterminal are basic symbols, they simply store a python string
# * Span is a composed symbol, it contains a Symbol and a range represented as two integers
# * Internally a Span is a python tuple of the kind (symbol: Symbol, start: int, end: int)
# * We define two *3* special methods to interact with basic and composed symbols
#     * root: goes all the way up to the root symbol (for example, returns the Symbol in a Span)
#     * obj: returns the underlying python object (for example, a str for Terminal, or tuple for Span)    
#     * translate: creates a symbol identical in structure, but translates the underlying python object of the root symbol (for example, translates the Terminal of a Span)
# 

class Symbol:
    """
    A symbol in a grammar. In this class we basically wrap a certain type of object and treat it as a symbol.
    """
    
    def __init__(self):
        pass
    
    def is_terminal(self) -> bool:
        """Whether or not this is a terminal symbol"""
        pass

    def root(self) -> 'Symbol':
        """Some symbols are represented as a hierarchy of symbols, this method returns the root of that hierarchy."""
        pass    
    
    def obj(self) -> object:
        """Returns the underlying python object."""
        pass
    
    def translate(self, target) -> 'Symbol':
        """Translate the underlying python object of the root symbol and return a new Symbol"""
        pass
    
class Terminal(Symbol):
    """
    Terminal symbols are words in a vocabulary.
    """
    
    def __init__(self, symbol: str):
        assert type(symbol) is str, 'A Terminal takes a python string, got %s' % type(symbol)
        self._symbol = symbol
        
    def is_terminal(self):
        return True
        
    def root(self) -> 'Terminal':
        # Terminals are not hierarchical symbols
        return self
    
    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol
    
    def translate(self, target) -> 'Terminal':
        return Terminal(target)
        
    def __str__(self):
        return "'%s'" % self._symbol
    
    def __repr__(self):
        return 'Terminal(%r)' % self._symbol
    
    def __hash__(self):
        return hash(self._symbol)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol
    
    def __ne__(self, other):
        return not (self == other)
    
class Nonterminal(Symbol):
    """
    Nonterminal symbols are variables in a grammar.
    """
    
    def __init__(self, symbol: str):
        assert type(symbol) is str, 'A Nonterminal takes a python string, got %s' % type(symbol)
        self._symbol = symbol
        
    def is_terminal(self):
        return False
        
    def root(self) -> 'Nonterminal':
        # Nonterminals are not hierarchical symbols
        return self
    
    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol
    
    def translate(self, target) -> 'Nonterminal':
        return Nonterminal(target)
    
    def __str__(self):
        return "[%s]" % self._symbol
    
    def __repr__(self):
        return 'Nonterminal(%r)' % self._symbol
    
    def __hash__(self):
        return hash(self._symbol)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol
    
    def __ne__(self, other):
        return not (self == other)
    


# The notion of *span* will come in handy when designing parsers, thus let's define it here.

class Span(Symbol):
    """
    A span can be a terminal, a nonterminal, or a span wrapped around two integers.
    Internally, we represent spans with tuples of the kind (symbol, start, end).
    
    Example:
        Span(Terminal('the'), 0, 1)
        Span(Nonterminal('[X]'), 0, 1)
        Span(Span(Terminal('the'), 0, 1), 1, 2)
        Span(Span(Nonterminal('[X]'), 0, 1), 1, 2)
    """
    
    def __init__(self, symbol: Symbol, start: int, end: int):
        assert isinstance(symbol, Symbol), 'A span takes an instance of Symbol, got %s' % type(symbol)
        self._symbol = symbol
        self._start = start
        self._end = end
        
    def is_terminal(self):
        # a span delegates this to an underlying symbol
        return self._symbol.is_terminal()
        
    def root(self) -> Symbol:
        # Spans are hierarchical symbols, thus we delegate 
        return self._symbol.root()
    
    def obj(self) -> (Symbol, int, int):
        """The underlying python tuple (Symbol, start, end)"""
        return (self._symbol, self._start, self._end)
    
    def translate(self, target) -> 'Span':
        return Span(self._symbol.translate(target), self._start, self._end)
    
    def __str__(self):
        return "%s:%s-%s" % (self._symbol, self._start, self._end)
    
    def __repr__(self):
        return 'Span(%r, %r, %r)' % (self._symbol, self._start, self._end)
    
    def __hash__(self):
        return hash((self._symbol, self._start, self._end))
    
    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol and self._start == other._start and self._end == other._end
    
    def __ne__(self, other):
        return not (self == other)


# # Rules 
# 
# 
# A context-free rule rewrites a nonterminal LHS symbol into a sequence of terminal and nonterminal symbols.
# We expect sequences to be non-empty, and we reserve a special terminal symbol to act as an epsilon string.

from collections import defaultdict

class Rule(object):
    """
    A rule is a container for a LHS symbol and a sequence of RHS symbols.
    """

    def __init__(self, lhs: Symbol, rhs: list):
        """
        A rule takes a LHS symbol and a list/tuple of RHS symbols
        """
        assert isinstance(lhs, Symbol), 'LHS must be an instance of Symbol'
        assert len(rhs) > 0, 'If you want an empty RHS, use an epsilon Terminal'
        assert all(isinstance(s, Symbol) for s in rhs), 'RHS must be a sequence of Symbol objects'
        self._lhs = lhs
        self._rhs = tuple(rhs)

    def __eq__(self, other):
        return type(self) == type(other) and self._lhs == other._lhs and self._rhs == other._rhs

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self._lhs, self._rhs))

    def __str__(self):
        return '%s ||| %s' % (self._lhs, ' '.join(str(s) for s in self._rhs))
    
    def __repr__(self):
        return 'Rule(%r, %r)' % (self._lhs, self._rhs)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs
    
    @property
    def arity(self):
        return len(self._rhs)
    


# # CFG
# 
# Now let us write a CFG class, which will organise rules for us in a convenient manner.
# We will design CFGs to be immutable.

class CFG:
    """
    A CFG is nothing but a container for rules.
    We group rules by LHS symbol and keep a set of terminals and nonterminals.
    """

    def __init__(self, rules=[]):
        self._rules = []
        self._rules_by_lhs = defaultdict(list)
        self._terminals = set()
        self._nonterminals = set()
        # organises rules
        for rule in rules:
            self._rules.append(rule)
            self._rules_by_lhs[rule.lhs].append(rule)
            self._nonterminals.add(rule.lhs)
            for s in rule.rhs:
                if s.is_terminal():
                    self._terminals.add(s)
                else:
                    self._nonterminals.add(s)

    @property
    def nonterminals(self):
        return self._nonterminals

    @property
    def terminals(self):
        return self._terminals

    def __len__(self):
        return len(self._rules)

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def get(self, lhs, default=frozenset()):
        """rules whose LHS is the given symbol"""
        return self._rules_by_lhs.get(lhs, default)

    def can_rewrite(self, lhs):
        """Whether a given nonterminal can be rewritten.

        This may differ from ``self.is_nonterminal(symbol)`` which returns whether a symbol belongs
        to the set of nonterminals of the grammar.
        """
        return len(self[lhs]) > 0

    def __iter__(self):
        """iterator over rules (in arbitrary order)"""
        return iter(self._rules)

    def items(self):
        """iterator over pairs of the kind (LHS, rules rewriting LHS)"""
        return self._rules_by_lhs.items()
    
    def iter_rules(self, lhs: Symbol):
        return iter(self.get(lhs))

    def __str__(self):
        lines = []
        for lhs, rules in self.items():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)


# # ITG
# 
# We do not really need a special class for ITGs, they are just a generalisation of CFGs for multiple streams.
# What we can do is to treat the source side and the target side of the ITG as CFGs.
# 
# We will represent a lexicon
# 
# * a collection of translation pairs \\((x, y) \in \Sigma \times \Delta\\) where \\(\Sigma\\) is the source vocabulary and \\(\Delta\\) is the target vocabulary
# * these vocabularies are extended with an empty string, i.e., \\(\epsilon\\)
# * we will assume the lexicon expliclty states which words can be inserted/deleted 
# 
# We build the source side by inspecting a lexicon
# 
# * terminal rules: \\(X \rightarrow x\\) where \\(x \in \Sigma\\)
# * binary rules: \\(X \rightarrow X ~ X\\)
# * start rule: \\(S \rightarrow X\\)
# 
# Then, when the time comes, we will project this source grammar using the lexicon
# 
# * terminal rules of the form \\(X_{i,j} \rightarrow x\\) will become \\(X_{i,j} \rightarrow y\\) for every possible translation pair \\((x, y)\\) in the lexicon
# * binary rules of the form \\(X_{i,k} \rightarrow X_{i,j} ~ X_{j,k}\\) will be copied and also inverted as in \\(X_{i,k} \rightarrow X_{j,k} ~ X_{i,j}\\)
# * the start rule will be copied
# 

def read_lexicon(path):
    """
    Read translation dictionary from a file (one word pair per line) and return a dictionary
    mapping x \in \Sigma to a set of y \in \Delta
    """
    lexicon = defaultdict(set)
    with open(path) as istream:        
        for n, line in enumerate(istream):
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) != 2:
                raise ValueError('I expected a word pair in line %d, got %s' % (n, line))
            x, y = words
            lexicon[x].add(y)
    return lexicon
            
def make_source_side_itg(lexicon, s_str='S', x_str='X') -> CFG:
    """Constructs the source side of an ITG from a dictionary"""
    S = Nonterminal(s_str)
    X = Nonterminal(x_str)
    def iter_rules():
        yield Rule(S, [X])  # Start: S -> X
        yield Rule(X, [X, X])  # Segment: X -> X X
        for x in lexicon.keys():
            yield Rule(X, [Terminal(x)])  # X - > x  
    return CFG(iter_rules())



# We typically represent sentences using finite-state automata, this allows for a more general view of parsing.
# Let's define an FSA class and a function to instantiate the FSA that corresponds to a sentence.

class FSA:
    """
    A container for arcs. This implements a deterministic unweighted FSA.
    """
    
    def __init__(self):
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        self._states = []  # here we map from origin to label to destination
        self._initial = set()
        self._final = set()
        self._arcs = []  # here we map from origin to destination to label
        
    def nb_states(self):
        """Number of states"""
        return len(self._states)
    
    def nb_arcs(self):
        """Number of arcs"""
        return sum(len(outgoing) for outgoing in self._states)
    
    def add_state(self, initial=False, final=False) -> int:
        """Add a state marking it as initial and/or final and return its 0-based id"""
        sid = len(self._states)
        self._states.append(defaultdict(int))
        self._arcs.append(defaultdict(str))
        if initial:
            self.make_initial(sid)
        if final:
            self.make_final(sid)
        return sid
    
    def add_arc(self, origin, destination, label: str):
        """Add an arc between `origin` and `destination` with a certain label (states should be added before calling this method)"""
        self._states[origin][label] = destination
        self._arcs[origin][destination] = label
    
    def destination(self, origin: int, label: str) -> int:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin >= len(self._states):
            return -1
        outgoing = self._states[origin] 
        if not outgoing:
            return -1
        return outgoing.get(label, -1)

    def label(self, origin: int, destination: int) -> str:
        """Return the label of an arc or None if the arc does not exist"""
        if origin >= len(self._arcs):
            return None
        return self._arcs[origin].get(destination, None)
    
    def make_initial(self, state: int):
        """Mark a state as initial"""
        self._initial.add(state)
        
    def is_initial(self, state: int) -> bool:
        """Test whether a state is initial"""
        return state in self._initial
        
    def make_final(self, state: int):
        """Mark a state as final/accepting"""
        self._final.add(state)
        
    def is_final(self, state: int) -> bool:
        """Test whether a state is final/accepting"""
        return state in self._final
        
    def iterinitial(self):
        """Iterates over initial states"""
        return iter(self._initial)
    
    def iterfinal(self):
        """Iterates over final states"""
        return iter(self._final)
    
    def iterarcs(self, origin: int):
        return self._states[origin].items() if origin < len(self._states) else []
    
    def __str__(self):
        lines = ['states=%d' % self.nb_states(), 
                 'initial=%s' % ' '.join(str(s) for s in self._initial),
                 'final=%s' % ' '.join(str(s) for s in self._final),
                 'arcs=%d' % self.nb_arcs()]        
        for origin, arcs in enumerate(self._states):
            for label, destination in sorted(arcs.items(), key=lambda pair: pair[1]):            
                lines.append('origin=%d destination=%d label=%s' % (origin, destination, label))
        return '\n'.join(lines)
        
def make_fsa(string: str) -> FSA:
    """Converts a sentence (string) to an FSA (labels are python str objects)"""
    fsa = FSA()
    fsa.add_state(initial=True)
    for i, word in enumerate(string.split()):
        fsa.add_state()  # create a destination state 
        fsa.add_arc(i, i + 1, word)  # label the arc with the current word
    fsa.make_final(fsa.nb_states() - 1)
    return fsa


# # Deductive system
# 
# We implement our parser using a deductive system.
# 
# ## Items
# 
# First we represent the items of our deductive system (again immutable objects).


"""
An item in a CKY/Earley program.
"""

class Item:
    """A dotted rule used in CKY/Earley where dots store the intersected FSA states."""

    def __init__(self, rule: Rule, dots: list):
        assert len(dots) > 0, 'I do not accept an empty list of dots'
        self._rule = rule
        self._dots = tuple(dots)

    def __eq__(self, other):
        return type(self) == type(other) and self._rule == other._rule and self._dots == other._dots

    def __ne__(self, other):
        return not(self == other)

    def __hash__(self):
        return hash((self._rule, self._dots))

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

from collections import defaultdict, deque

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
        self._complete = defaultdict(set)
        # here we store the destinations already discovered
        self._destinations = defaultdict(set)

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
            self._complete[(item.lhs, item.start)].add(item)
            self._destinations[(item.lhs, item.start)].add(item.dot)
        else:  # incomplete items are waiting for the completion of the symbol to the right of the dot
            self._incomplete[(item.next, item.dot)].add(item)

    def waiting(self, symbol: Symbol, dot: int) -> set:
        """return items waiting for `symbol` spanning from `dot`"""
        return self._incomplete.get((symbol, dot), set())

    def complete(self, lhs: Symbol, start: int) -> set:
        """return complete items whose LHS symbol is `lhs` spanning from `start`"""
        return self._complete.get((lhs, start), set())
    
    def destinations(self, lhs: Symbol, start: int) -> set:
        """return destinations (in the FSA) for `lhs` spanning from `start`"""
        return self._destinations.get((lhs, start), set())

    def itercomplete(self):
        """an iterator over complete items in arbitrary order"""
        for items in self._complete.itervalues():
            for item in items:
                yield item


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
    items = []
    for rule in cfg.get(item.next):
        items.append(Item(rule, [item.dot]))
    return items

def scan(fsa: FSA, item: Item, eps_symbol: Terminal=Terminal('-EPS-')) -> list:
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
    if eps_symbol and item.next.root() == eps_symbol:
        return [item.advance(item.dot)]
    else:
        destination = fsa.destination(origin=item.dot, label=item.next.root().obj())  # we call .obj() because labels are strings, not Terminals
        if destination < 0:  # cannot scan the symbol from this state
            return []
        return [item.advance(destination)]
        
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
    items = []
    if item.is_complete():
        # advance the dot for incomplete items waiting for item.lhs spanning from item.start
        for incomplete in agenda.waiting(item.lhs, item.start):
            items.append(incomplete.advance(item.dot))
    else:
        # look for completions of item.next spanning from item.dot
        for destination in agenda.destinations(item.next, item.dot):                
            items.append(item.advance(destination))
    return items
    
def earley(cfg: CFG, fsa: FSA, start_symbol: Symbol, sprime_symbol=None, eps_symbol=Terminal('-EPS-')):
    """
    Earley intersection between a CFG and an FSA.
    
    :param cfg: a grammar or forest
    :param fsa: an acyclic FSA
    :param start_symbol: the grammar/forest start symbol
    :param sprime_symbol: if specified, the resulting forest will have sprime_symbol as its starting symbol
    :param eps_symbol: if not None, the parser will support epsilon rules
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
            for item in A.complete(nonterminal, start):
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
    return CFG(iter_intersected_rules())


# # Target side of the ITG
# 
# Now we can project the forest onto the target vocabulary by using ITG rules.

def make_target_side_itg(source_forest: CFG, lexicon: dict) -> CFG:
    """Constructs the target side of an ITG from a source forest and a dictionary"""    
    def iter_rules():
        for lhs, rules in source_forest.items():            
            for r in rules:
                if r.arity == 1:  # unary rules
                    if r.rhs[0].is_terminal():  # terminal rules
                        x_str = r.rhs[0].root().obj()  # this is the underlying string of a Terminal
                        targets = lexicon.get(x_str, set())
                        if not targets:
                            pass  # TODO: do something with unknown words?
                        else:
                            for y_str in targets:
                                yield Rule(r.lhs, [r.rhs[0].translate(y_str)])  # translation
                    else:
                        yield r  # nonterminal rules
                elif r.arity == 2:
                    yield r  # monotone
                    if r.rhs[0] != r.rhs[1]:  # avoiding some spurious derivations by blocking invertion of identical spans
                        yield Rule(r.lhs, [r.rhs[1], r.rhs[0]])  # inverted
                else:
                    raise ValueError('ITG rules are unary or binary, got %r' % r)        
    return CFG(iter_rules())


# # Legth constraint
# 
# To constrain the space of derivations by length we can parse a special FSA using the forest that represents \\(D(x)\\), i.e. `tgt_forest` in the code above.
# 
# For maximum lenght \\(n\\), this special FSA must accept the language \\(\Sigma^0 \cup \Sigma^1 \cup \cdots \cup \Sigma^n\\). You can implement this FSA designing a special FSA class which never rejects a terminal (for example by defining a *wildcard* symbol).
# 


class LengthConstraint(FSA):
    """
    A container for arcs. This implements a deterministic unweighted FSA.
    """
    
    def __init__(self, n: int, strict=False):
        """
        :param n: length constraint
        :param strict: if True, accepts the language \Sigma^n, if False, accepts union of \Sigma^i for i from 0 to n
        """
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        super(LengthConstraint, self).__init__()
        assert n > 0, 'We better use n > 0.'
        self.add_state(initial=True)  # we start by adding an initial state
        for i in range(n):
            self.add_state(final=not strict)  # then we add a state for each unit of length
            self.add_arc(i, i + 1, '-WILDCARD-')  # and an arc labelled with a WILDCARD
        # we always make the last state final
        self.make_final(n)
                
    def destination(self, origin: int, label: str) -> int:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin + 1 < self.nb_states():
            outgoing = self._states[origin] 
            if not outgoing:
                return -1
            return origin + 1
        else:
            return -1

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
    accepting = set()
    for rule in forest.iter_rules(start_symbol):  # S' -> S:initial-final
        for sym in rule.rhs:  # the RHS contains of top rules contain the accepting states 
            s, initial, final = sym.obj()
            accepting.add(final)
   
    def visit_forest_node(symbol: Symbol, bos, eos, parent: Symbol):
        """Visit a symbol spanning from bos to eos given a parent symbol"""
        if symbol.is_terminal():
            fsa.add_arc(bos, eos, symbol.root().obj())
            if isinstance(parent, Span) and parent.obj()[-1] in accepting:
                fsa.make_final(eos)
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


def enumerate_paths_in_fsa(fsa: FSA, eps_str='-EPS-') -> set:
    # then we enumerate paths in this FSA
    paths = set()
    
    def visit_fsa_node(state, path):
        if fsa.is_final(state):
            paths.add(' '.join(x for x in path if x != eps_str))
        for label, destination in fsa.iterarcs(state):
            visit_fsa_node(destination, path + [label])
    
    visit_fsa_node(0, [])
    
    return paths


