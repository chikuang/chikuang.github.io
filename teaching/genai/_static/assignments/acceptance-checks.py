"""Independent, hand-computable acceptance fixtures; no reference answers imported.

These check the requested operation, not the quality of an experimental design.
"""
import numpy as np


def check(chapter, function):
    f = function
    if chapter == 1:
        assert np.isclose(f([4,1],[.6,.4]), .2)
        assert np.isclose(f([1,3],[.5,.5]), .25)
        assert np.isclose(f([4,0],[1,0]), 0)
    elif chapter == 2:
        for beta, expected_loss, expected_grad in [(0., np.log(2), -1.), (np.log(3)/2, -np.log(.75), -.5)]:
            loss, gradient=f(np.array([[2.]]),np.array([1.]),np.array([beta]))
            assert np.isclose(loss,expected_loss)
            assert gradient.shape == (1,) and np.allclose(gradient,[expected_grad])
        loss, gradient=f(np.ones((2,1)), np.array([0.,1.]), np.zeros(1))
        assert np.isclose(loss,np.log(2)) and np.allclose(gradient,[0.])
    elif chapter == 3:
        assert f(['a','b'],['b','c'],2) == .5
        assert f(['a','b'],['a','b'],2) == 1
        assert f(['a','b'],['c','d'],2) == 0
        assert f(['a','b','c'],['a','c','b'],1) == 1
    elif chapter == 4:
        s=np.array([[0.,999.],[0.,np.log(3)]])
        w,y=f(s,np.array([3.,9.]))
        assert np.allclose(w,[[1.,0.],[.25,.75]]) and np.allclose(y,[3.,7.5])
        w2,y2=f(s+1000,np.array([3.,9.]))
        assert np.allclose(w2,w) and np.allclose(y2,y)
    elif chapter == 5:
        before=np.arange(12.).reshape(1,3,4)
        after=before.copy();after[:,-1]+=2
        assert f(before,after,2)
        assert not f(before,after,3)
        after[:,0,0]+=1
        assert not f(before,after,2)
    elif chapter == 6:
        assert f({'text':'a b','stop_reason':'end_token'},{'a b'})
        assert not f({'text':'wrong','stop_reason':'end_token'},{'a b'})
        assert not f({'text':'a b','stop_reason':'length_limit'},{'a b'})
    elif chapter == 7:
        samples=np.array([[0.,0.],[3.,4.]])
        centers=np.array([[0.,0.]])
        assert f(samples,centers,0) == .5
        assert f(samples,centers,4.9) == .5
        assert f(samples,centers,5) == 1
        assert f(np.array([[3.,4.]]),centers,4) == 0
    elif chapter == 8:
        assert np.allclose(f(np.array([-1.,0.,3.]),1.,4.),[.5,.25,-.5])
        assert np.allclose(f(np.array([-1.,0.,1.]),0.,1.),[1.,0.,-1.])
    elif chapter == 9:
        assert f(12,8,2) == 40
        assert f(3,5,1) == 8
        assert f(3,5,2) == 16
    elif chapter == 10:
        valid={'answer':'Use blocks.','citations':['s1'],'abstain':False}
        assert f(valid,{'s1'})
        assert not f(dict(valid,answer=''),{'s1'})
        assert not f(dict(valid,citations=[]),{'s1'})
        assert not f(dict(valid,citations=['s1','s1']),{'s1'})
        assert not f(dict(valid,citations=['s2']),{'s1'})
        assert not f(dict(valid,abstain=1),{'s1'})
        assert f({'answer':'Insufficient evidence.','citations':[],'abstain':True},{'s1'})
        # Structural acceptance is not semantic support.
        assert f(dict(valid,answer='Dependence never matters.'),{'s1'})
    elif chapter == 11:
        assert np.array_equal(f([True,False,True],[False,True,True]),[-1.,1.,0.])
        for a,b in [([],[]),([1],[1,0]),([[1]],[[1]]),([np.nan],[0])]:
            try:f(a,b)
            except ValueError:pass
            else:raise AssertionError('Invalid paired scores accepted')
    elif chapter == 12:
        state={'observed':{},'results':[{'value':1.5}]}
        assert f(1.5,state)
        assert not f(999,state)
        assert not f(1.5,{'observed':{},'results':[]})
        assert not f(True,{'observed':{},'results':[{'value':1.0}]})
        assert not f(float('nan'),state)
    elif chapter == 13:
        assert np.isclose(f({(1,False):.4,(3,False):.6,(1,True):.5,(3,True):.9}),.2)
        assert np.isclose(f({(1,False):.2,(3,False):.4,(1,True):.3,(3,True):.5}),0)
    elif chapter == 14:
        card=dict(family='normal',n=18,method='z_estimated_sd',confidence=.95,repetitions=500,dependence='iid')
        assert f(card,card)
        assert not f(card,None)
        assert not f(dict(card,family='exponential'),card)
        for key,value in [('n',True),('method','student_t'),('confidence',.99),('repetitions',2000),('dependence','dependent'),('family','cauchy')]:
            changed=dict(card,**{key:value})
            assert not f(changed,changed), key
        assert not f({'family':'normal','n':18},card)
    else:
        raise ValueError('Chapter must be 1 through 14')
    return f'Chapter {chapter}: numerical/contract fixtures passed; experiment and interpretation still need review.'
