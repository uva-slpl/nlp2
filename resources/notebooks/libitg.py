"""
Here we implement ITG-related algorithms to fit a LV-CRF.

:author: Wilker Aziz
"""

from formal import *
from alg import * 
from earley import *
from time import time
import numpy as np
import sys


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

        
def make_fsa(string: str) -> FSA:
    """Converts a sentence (string) to an FSA (labels are python str objects)"""
    fsa = FSA()
    fsa.add_state(initial=True)
    for i, word in enumerate(string.split()):
        fsa.add_state()  # create a destination state 
        fsa.add_arc(i, i + 1, word)  # label the arc with the current word
    fsa.make_final(fsa.nb_states() - 1)
    return fsa


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
    This implement an automaton that accepts strings containing up to n (non-empty) symbols.
    """
    
    def __init__(self, n: int, strict=False, wildcard_str='-WILDCARD-'):
        """
        :param n: length constraint
        :param strict: if True, accepts the language \Sigma^n, if False, accepts union of \Sigma^i for i from 0 to n
        """
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        super(LengthConstraint, self).__init__()
        assert n > 0, 'We better use n > 0.'
        self.add_state(initial=True, final=not strict)  # we start by adding an initial state
        for i in range(n):
            self.add_state(final=not strict)  # then we add a state for each unit of length
            self.add_arc(i, i + 1, wildcard_str)  # and an arc labelled with a WILDCARD
        # we always make the last state final
        self.make_final(n)
        self._wildcard_str = wildcard_str
                
    def destinations(self, origin: int, label: str) -> set:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin < self.nb_states():
            return super(LengthConstraint, self).destinations(origin, self._wildcard_str)
        else:
            return set()


class InsertionConstraint(FSA):
    """
    This implements an automaton that accepts up to n insertions.

    For this you need to make Earley think that -EPS- is a normal terminal,
        you can do that by setting eps_symbol to None when calling earley.
    """
    
    def __init__(self, n: int, strict=False, eps_str='-EPS-', wildcard_str='-WILDCARD-'):
        """
        :param n: length constraint
        :param strict: if True, accepts exactly n insertions, if False, accepts up to n insertions.
        """
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        super(InsertionConstraint, self).__init__()
        assert n >=0 , 'We better use n > 0.'
        self.add_state(initial=True, final=not strict)  # we start by adding an initial state
        self.add_arc(0, 0, wildcard_str)
        for i in range(n):
            self.add_state(final=not strict)  # then we add a state for each unit of length
            self.add_arc(i, i + 1, eps_str)  # and an arc labelled with a WILDCARD
            self.add_arc(i + 1, i + 1, wildcard_str)  # and an arc labelled with a WILDCARD
        # we always make the last state final
        self.make_final(n)
        self._eps_str = eps_str
        self._wildcard_str = wildcard_str
                
    def destinations(self, origin: int, label: str) -> set:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin < self.nb_states():
            if label == self._eps_str:
                return super(InsertionConstraint, self).destinations(origin, label)
            else:  # if not eps, we match any word
                return super(InsertionConstraint, self).destinations(origin, self._wildcard_str)
        else:
            return set()


def summarise_strings(forest, root):
    strings = language_of_fsa(forest_to_fsa(forest, root))
    print(' Strings from %s: %d' % (root, len(strings)))
    #for string in sorted(strings, key=lambda v: (len(v), v)):
    #    print(' ',string)
    return strings


def test(lexicon, src_str, tgt_str, constraint_type='length', nb_insertions=0, inspect_strings=False):

    print('TRAINING INSTANCE: |x|=%d |y|=%d' % (len(src_str.split()), len(tgt_str.split())))
    print(src_str)
    print(tgt_str)
    print()

    # Make a source CFG using the whole lexicon
    src_cfg = make_source_side_itg(lexicon)
    #print('SOURCE CFG')
    #print(src_cfg)
    #print()

    
    # Make a source FSA
    src_fsa = make_fsa(src_str)
    #print('SOURCE FSA')
    #print(src_fsa)
    #print()
    # Make a target FSA
    tgt_fsa = make_fsa(tgt_str)
    #print('TARGET FSA')
    #print(tgt_fsa)

    # Intersect source FSA and source CFG
    times = dict()
    times['D(x)'] = time()
    _Dx = earley(src_cfg, src_fsa, 
            start_symbol=Nonterminal('S'), 
            sprime_symbol=Nonterminal("D(x)"),
            clean=False)  # to illustrate the difference between clean and dirty forests I will disable clean here
    #print(src_forest)
    #print()
    # projection over target vocabulary
    Dx = make_target_side_itg(_Dx, lexicon)
    times['D(x)'] = time() - times['D(x)']
    
    times['clean D(x)'] = time()
    Dx_clean = cleanup_forest(Dx, Nonterminal('D(x)'))
    times['clean D(x)'] = time() - times['clean D(x)']

    if constraint_type == 'length':
        # we need to accept the length of the input
        print('Using LengthConstraint')
        constraint_fsa = LengthConstraint(tgt_fsa.nb_states() - 1, strict=False)        
        print(constraint_fsa)
        # in this case we constrain after projection (so that we can discount deletions and count insertions)
        times['D_n(x)'] = time()
        Dnx = earley(Dx, constraint_fsa,
                start_symbol=Nonterminal('D(x)'),
                sprime_symbol=Nonterminal('D_n(x)'),
                clean=False)  
        times['D_n(x)'] = time() - times['D_n(x)']
        # here we parse observation y with D(x)
        #  because we choose the length such that n >= |y| and thus we are sure that y \in D_n(x)
        times['D(x,y)'] = time()
        Dxy = earley(Dx, tgt_fsa,
                start_symbol=Nonterminal("D(x)"), 
                sprime_symbol=Nonterminal('D(x,y)'),
                clean=False)
        times['D(x,y)'] = time() - times['D(x,y)']
    else:
        print('Using InsertionConstraint')
        constraint_fsa = InsertionConstraint(nb_insertions, strict=False)
        print(constraint_fsa)
        # in this case we constrain before projection (so we can count epsilon transitions)
        times['D_n(x)'] = time()
        _Dnx = earley(_Dx, constraint_fsa,
                start_symbol=Nonterminal('D(x)'),
                sprime_symbol=Nonterminal('D_n(x)'),
                eps_symbol=None,
                clean=False)  # for this we make eps count as if it were a real symbol
        Dnx = make_target_side_itg(_Dnx, lexicon)
        times['D_n(x)'] = time() - times['D_n(x)']
        # here we parse observation y using D_n(x)
        #  because there is no guarantee that y \in D_n(x) and we need to be sure
        times['D(x,y)'] = time()
        Dxy = earley(Dnx, tgt_fsa,
                start_symbol=Nonterminal("D_n(x)"), 
                sprime_symbol=Nonterminal('D(x,y)'),
                clean=False)
        times['D(x,y)'] = time() - times['D(x,y)']

    times['clean D_n(x)'] = time()
    Dnx_clean = cleanup_forest(Dnx, Nonterminal('D_n(x)'))
    times['clean D_n(x)'] = time() - times['clean D_n(x)']
    
    times['clean D(x,y)'] = time()
    Dxy_clean = cleanup_forest(Dxy, Nonterminal('D(x,y)'))
    times['clean D(x,y)'] = time() - times['clean D(x,y)']
    
    print('D(x): %d rules in %.4f secs or clean=%d rules at extra %.4f secs' % (len(Dx), times['D(x)'],
        len(Dx_clean), times['clean D(x)']))
    print('D_n(x): %d rules in %.4f secs or clean=%d rules at extra %.4f secs' % (len(Dnx), 
        times['D_n(x)'], len(Dnx_clean), times['clean D_n(x)']))
    print('D(x,y): %d rules in %.4f secs or clean=%d rules at extra %.4f secs' % (len(Dxy), 
        times['D(x,y)'], len(Dxy_clean), times['clean D(x,y)']))
    
    if inspect_strings:
        t0 = time()
        print(' y in D_n(x):', tgt_str in summarise_strings(Dnx, Nonterminal('D_n(x)')))
        print(' y in clean D_n(x):', tgt_str in summarise_strings(Dnx_clean, Nonterminal('D_n(x)')))
        print(' y in D(x,y):', tgt_str in summarise_strings(Dxy, Nonterminal('D(x,y)')))
        print(' y in clean D(x,y):', tgt_str in summarise_strings(Dxy_clean, Nonterminal('D(x,y)')))
        print(' gathering strings took %d secs' % (time() - t0))

    print()

if __name__ == '__main__':
    # Test lexicon
    lexicon = defaultdict(set)
    lexicon['le'].update(['-EPS-', 'the', 'some', 'a', 'an'])  # we will assume that `le` can be deleted
    lexicon['e'].update(['-EPS-', 'and', '&', 'also', 'as'])
    lexicon['chien'].update(['-EPS-', 'dog', 'canine', 'wolf', 'puppy'])
    lexicon['noir'].update(['-EPS-', 'black', 'noir', 'dark', 'void'])  
    lexicon['blanc'].update(['-EPS-', 'white', 'blank', 'clear', 'flash'])
    lexicon['petit'].update(['-EPS-', 'small', 'little', 'mini', 'junior'])
    lexicon['petite'].update(['-EPS-', 'small', 'little', 'mini', 'junior'])
    lexicon['.'].update(['-EPS-', '.', '!', '?', ','])
    
    #lexicon['-EPS-'].update(['.', ',', 'a', 'the', 'some', 'of', 'bit', "'s", "'m", "'ve"])  # we will assume that `the` and `a` can be inserted
    lexicon['-EPS-'].update(['.', 'a', 'the', 'some', 'of'])  # we will assume that `the` and `a` can be inserted
    print('LEXICON')
    for src_word, tgt_words in lexicon.items():
        print('%s: %d options' % (src_word, len(tgt_words)))
    print()
    
    test(lexicon, 
            'le chien noir',
            'black dog',
            'length', inspect_strings=True)
    test(lexicon, 
            'le chien noir',
            'the black dog .',
            'insertion', nb_insertions=1, inspect_strings=True)
    test(lexicon,
            'le petit chien noir e le petit chien blanc .',
            'the little white dog and the little black dog .',
            'length')
    test(lexicon,
            'le petit chien noir e le petit chien blanc .',
            'the little white dog and the little black dog .',
            'insertion', nb_insertions=3)

    sys.exit()
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien .', 
            'the little black dog and the little white dog and the mini dog .',
            'length')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien .', 
            'the little black dog and the little white dog and the mini dog .',
            'insertion', nb_insertions=3)
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien petit blanc e petit noir .', 
            'the little black dog and the little white dog and the dog a bit white and a bit black .',
            'length')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien petit blanc e petit noir .', 
            'the little black dog and the little white dog and the dog a bit white and a bit black .',
            'insertion', nb_insertions=3)

