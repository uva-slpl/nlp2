"""
Here I define a number of immutable objects that will be useful in designing the parser

* Symbol
    * Terminal
    * Nonterminal
    * Span
* Rule
* Item

They all manage instances of their classes in a way that we are guaranteed to have one instance per object.
This allows us to design more efficient comparison and hash functions.
This also saves memory.

:author: Wilker Aziz
"""

from weakref import WeakValueDictionary
from collections import defaultdict


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

    __repositories = defaultdict(WeakValueDictionary)

    def __new__(cls, key, constructor):
        repository = Symbol.__repositories[cls]
        instance = repository.get(key, None)
        if instance is None:
            instance = constructor()
            repository[key] = instance
        return instance

    @classmethod
    def nb_instances(cls):
        """Number of instances of cls (or total for all classes if cls is Symbol)"""
        if cls is Symbol:
            return sum(len(repo) for repo in Symbol.__repositories.values())
        else:
            return len(Symbol.__repositories.get(cls, {}))

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    def is_terminal(self) -> bool:
        """Whether or not this is a terminal symbol"""
        pass

    def root(self) -> 'Symbol':
        """Some symbols are represented as a hierarchy of symbols, this method returns the root of that hierarchy."""
        pass    
    
    def obj(self) -> object:
        """Returns the underlying python object."""
        pass
    
    def translate(self, target: str) -> 'Symbol':
        """Translate the underlying python string of the root symbol and return a new Symbol"""
        pass
    
class Terminal(Symbol):
    """
    Terminal symbols are words in a vocabulary.
    """
    
    def __new__(cls, symbol: str):
        assert type(symbol) is str, 'A Terminal takes a python string, got %s' % type(symbol)
        def constructor():
            instance = object.__new__(Terminal)
            instance._symbol = symbol
            return instance
        return super(Terminal, cls).__new__(cls, symbol, constructor)
        
    def is_terminal(self):
        return True
        
    def root(self) -> 'Terminal':
        # Terminals are not hierarchical symbols
        return self
    
    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol
    
    def translate(self, target: str) -> 'Terminal':
        return Terminal(target)
        
    def __str__(self):
        return "'%s'" % self._symbol
    
    def __repr__(self):
        return 'Terminal(%r)' % self._symbol
    
    
class Nonterminal(Symbol):
    """
    Nonterminal symbols are variables in a grammar.
    """
    
    def __new__(cls, symbol: str):
        assert type(symbol) is str, 'A Nonterminal takes a python string, got %s' % type(symbol)
        def constructor():
            instance = object.__new__(Nonterminal)
            instance._symbol = symbol
            return instance
        return super(Nonterminal, cls).__new__(cls, symbol, constructor)
        
    def is_terminal(self):
        return False
        
    def root(self) -> 'Nonterminal':
        # Nonterminals are not hierarchical symbols
        return self
    
    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol
    
    def translate(self, target: str) -> 'Nonterminal':
        return Nonterminal(target)
    
    def __str__(self):
        return "[%s]" % self._symbol
    
    def __repr__(self):
        return 'Nonterminal(%r)' % self._symbol


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
    
    def __new__(cls, symbol: Symbol, start: int, end: int):
        assert isinstance(symbol, Symbol), 'A span takes an instance of Symbol, got %s' % type(symbol)
        def constructor():
            instance = object.__new__(Span)
            instance._symbol = symbol
            instance._start = start
            instance._end = end
            return instance
        return super(Span, cls).__new__(cls, (symbol, start, end), constructor)
        
    def is_terminal(self):
        # a span delegates this to an underlying symbol
        return self._symbol.is_terminal()
        
    def root(self) -> Symbol:
        # Spans are hierarchical symbols, thus we delegate 
        return self._symbol.root()
    
    def obj(self) -> (Symbol, int, int):
        """The underlying python tuple (Symbol, start, end)"""
        return (self._symbol, self._start, self._end)
    
    def translate(self, target: str) -> 'Span':
        return Span(self._symbol.translate(target), self._start, self._end)
    
    def __str__(self):
        return "%s:%s-%s" % (self._symbol, self._start, self._end)
    
    def __repr__(self):
        return 'Span(%r, %r, %r)' % (self._symbol, self._start, self._end)


class Rule(object):
    """
    A rule is a container for a LHS symbol and a sequence of RHS symbols.
    """

    __repositories = defaultdict(WeakValueDictionary)

    def __new__(cls, lhs: Symbol, rhs: list):
        """
        A rule takes a LHS symbol and a list/tuple of RHS symbols
        """
        assert isinstance(lhs, Symbol), 'LHS must be an instance of Symbol'
        assert len(rhs) > 0, 'If you want an empty RHS, use an epsilon Terminal'
        assert all(isinstance(s, Symbol) for s in rhs), 'RHS must be a sequence of Symbol objects'
        rhs = tuple(rhs)
        repository = Rule.__repositories[lhs]
        instance = repository.get(rhs, None)
        if instance is None:
            instance = object.__new__(Rule)
            instance._lhs = lhs
            instance._rhs = rhs
            repository[rhs] = instance
        return instance
    
    @classmethod
    def nb_instances(cls):
        """Number of instances of Rule"""
        return sum(len(repo) for repo in Rule.__repositories.values())

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

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
    
    def __eq__(self, other):
        return type(self) == type(other) and self._rules == other._rules 

    def __ne__(self, other):
        return not (self == other)

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


class FSA:
    """
    A container for arcs. This implements a non-deterministic unweighted FSA.
    """

    class State:
        def __init__(self):
            self.by_destination = defaultdict(set)
            self.by_label = defaultdict(set)

        def __eq__(self, other):
            return type(self) == type(other) and self.by_destination == other.by_destination and self.by_label == other.by_label

        def __ne__(self, other):
            return not (self == other)
    
    def __init__(self):
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        # each state is a tuple (by_destination and by_label)
        #  by_destination is a dictionary that maps from destination to a set of labels
        #  by_label is a dictionary that maps from label to a set of destinations
        self._states = [] 
        self._initial = set()
        self._final = set()
        self._arcs = set()
    
    def __eq__(self, other):
        return type(self) == type(other) and self._states == other._states and self._initial == other._initial and self._final == other._final

    def __ne__(self, other):
        return not (self == other)
        
    def nb_states(self):
        """Number of states"""
        return len(self._states)
    
    def nb_arcs(self):
        """Number of arcs"""
        return len(self._arcs)
    
    def add_state(self, initial=False, final=False) -> int:
        """Add a state marking it as initial and/or final and return its 0-based id"""
        sid = len(self._states)
        self._states.append(FSA.State())
        if initial:
            self.make_initial(sid)
        if final:
            self.make_final(sid)
        return sid
    
    def add_arc(self, origin, destination, label: str):
        """Add an arc between `origin` and `destination` with a certain label (states should be added before calling this method)"""
        self._states[origin].by_destination[destination].add(label)
        self._states[origin].by_label[label].add(destination)
        self._arcs.add((origin, destination, label))

    def destinations(self, origin: int, label: str) -> set:
        if origin >= len(self._states):
            return set()
        return self._states[origin].by_label.get(label, set())
    
    def labels(self, origin: int, destination: int) -> set:
        """Return the label of an arc or None if the arc does not exist"""
        if origin >= len(self._arcs):
            return set()
        return self._states[origin].by_destination.get(destination, set())
    
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
   
    def iterarcs(self, origin: int, group_by='destination') -> dict:
        if origin + 1 < self.nb_states():
            return self._states[origin].by_destination.items() if group_by == 'destination' else self._states[origin].by_label.items()
        return dict()
    
    def __str__(self):
        lines = ['states=%d' % self.nb_states(), 
                 'initial=%s' % ' '.join(str(s) for s in self._initial),
                 'final=%s' % ' '.join(str(s) for s in self._final),
                 'arcs=%d' % self.nb_arcs()]        
        for origin, state in enumerate(self._states):
            for destination, labels in sorted(state.by_destination.items(), key=lambda pair: pair[0]):            
                for label in sorted(labels):
                    lines.append('origin=%d destination=%d label=%s' % (origin, destination, label))
        return '\n'.join(lines)
        
