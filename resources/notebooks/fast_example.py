from formal import *
from alg import * 
from earley import *
from time import time
import libitg
import numpy as np
import sys


def test(lexicon, src_str, tgt_str, inspect_strings=False):

    print('TRAINING INSTANCE: |x|=%d |y|=%d' % (len(src_str.split()), len(tgt_str.split())))
    print(src_str)
    print(tgt_str)
    print()

    # Make a source CFG using the whole lexicon
    src_cfg = libitg.make_source_side_finite_itg(lexicon)
    #print('SOURCE CFG')
    #print(src_cfg)
    #print()

    
    # Make a source FSA
    src_fsa = libitg.make_fsa(src_str)
    #print('SOURCE FSA')
    #print(src_fsa)
    #print()
    # Make a target FSA
    tgt_fsa = libitg.make_fsa(tgt_str)
    #print('TARGET FSA')
    #print(tgt_fsa)

    # Intersect source FSA and source CFG
    times = dict()
    times['D(x)'] = time()
    _Dx = libitg.earley(src_cfg, src_fsa, 
            start_symbol=Nonterminal('S'), 
            sprime_symbol=Nonterminal("D(x)"),
            clean=False)  # to illustrate the difference between clean and dirty forests I will disable clean here
    #print(src_forest)
    #print()
    # projection over target vocabulary

    # D(x) is now finite :D
    Dx = libitg.make_target_side_finite_itg(_Dx, lexicon)
    times['D(x)'] = time() - times['D(x)']
    
    times['clean D(x)'] = time()
    Dx_clean = libitg.cleanup_forest(Dx, Nonterminal('D(x)'))
    times['clean D(x)'] = time() - times['clean D(x)']

    times['D(x,y)'] = time()
    Dxy = libitg.earley(Dx, tgt_fsa,
            start_symbol=Nonterminal("D(x)"), 
            sprime_symbol=Nonterminal('D(x,y)'),
            clean=False)
    times['D(x,y)'] = time() - times['D(x,y)']

    times['clean D(x,y)'] = time()
    Dxy_clean = libitg.cleanup_forest(Dxy, Nonterminal('D(x,y)'))
    times['clean D(x,y)'] = time() - times['clean D(x,y)']
    
    print('D(x): %d rules in %.4f secs or clean=%d rules at extra %.4f secs' % (len(Dx), times['D(x)'],
        len(Dx_clean), times['clean D(x)']))
    print('D(x,y): %d rules in %.4f secs or clean=%d rules at extra %.4f secs' % (len(Dxy), 
        times['D(x,y)'], len(Dxy_clean), times['clean D(x,y)']))

    
    if inspect_strings:
        t0 = time()
        print(' y in D(x):', tgt_str in libitg.summarise_strings(Dx, Nonterminal('D(x)')))
        print(' y in D(x,y):', tgt_str in libitg.summarise_strings(Dxy, Nonterminal('D(x,y)')))
        print(' y in clean D(x,y):', tgt_str in libitg.summarise_strings(Dxy_clean, Nonterminal('D(x,y)')))
        print(' gathering strings took %d secs' % (time() - t0))
    
    # and this is how you pickle things
    import dill as pickle
    with open('pickle-test', 'wb') as f:
        pickle.dump(Dxy_clean, f)
    with open('pickle-test', 'rb') as f:
        Dloaded = pickle.load(f)
    print(len(Dloaded), 'loaded')

    print()


if __name__ == '__main__':
    # Test lexicon
    lexicon = defaultdict(set)
    lexicon['le'].update(['-EPS-', 'the', 'some', 'a', 'an'])  # we will assume that `le` can be deleted
    lexicon['e'].update(['-EPS-', 'and', '&', 'also', 'as'])
    lexicon['chien'].update(['-EPS-', 'dog', 'canine', 'wolf', 'puppy'])
    lexicon['noir'].update(['-EPS-', 'black', 'noir', 'dark', 'void'])  
    lexicon['blanc'].update(['-EPS-', 'white', 'blank', 'clear', 'flash'])
    lexicon['petit'].update(['-EPS-', 'small', 'little', 'mini', 'almost'])
    lexicon['petite'].update(['-EPS-', 'small', 'little', 'mini', 'almost'])
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
            inspect_strings=False)
    test(lexicon, 
            'le chien noir',
            'the black dog .',
            inspect_strings=False)
    test(lexicon,
            'le petit chien noir e le petit chien blanc .',
            'the little white dog and the little black dog .')
    test(lexicon,
            'le petit chien noir e le petit chien blanc .',
            'the little white dog and the little black dog .')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien .', 
            'the little black dog and the little white dog and the mini dog .')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit chien petit blanc .', 
            'the little black dog and the little white dog and the mini almost white dog .')
    print('**** The next example should be out of the space of the constrained ITG ***')
    test(lexicon,
            'le petit chien noir e le petit chien blanc e le petit petit chien petit blanc e petit noir .', 
            'the little black dog and the little white dog and the dog a bit white and a bit black .')
