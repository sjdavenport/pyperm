"""
Testing the functions in the classes.py file
"""
import numpy as np
import pyrft as pr

def test_Field():
    """ Testing the Field class """
    # 1D Field
    nsubj = 30
    nvox = 100
    field = np.random.randn(nvox,nsubj)
    mask = np.ones((nvox,1), dtype = bool)
    f = pr.Field(field, mask)

    assert isinstance(f, pr.classes.Field)
    assert f.fieldsize == (nvox, nsubj)
    assert f.D == 1
    assert f.masksize == (1, nvox)
    assert f.fibersize == nsubj
    assert f.field.shape == f.fieldsize

    # 2D Field
    nsubj = 30
    dim = (100, 100)

    fsize = dim + (nsubj,)
    field = np.random.randn(*fsize)
    mask = np.ones(dim, dtype = bool)
    f = pr.Field(field, mask)

    assert isinstance(f, pr.classes.Field)
    assert f.fieldsize == dim + (nsubj,)
    assert f.D == len(dim)
    assert f.masksize == dim
    assert f.fibersize == nsubj
    assert f.field.shape == f.fieldsize

    # 2D field with a single fiber
    dim = (100,100)
    field = np.random.randn(*dim)
    mask = np.ones(dim, dtype = bool)
    f = pr.Field(field, mask)

    assert isinstance(f, pr.classes.Field)
    assert f.fieldsize == dim
    assert f.D == len(dim)
    assert f.masksize == dim
    assert f.fibersize == 1
    assert f.field.shape == f.fieldsize

def test_make_field():
    """ Testing the make_field function """
    nvox = 100
    nsubj = 30
    data = np.random.randn(nvox, nsubj)

    f = pr.make_field(data)
    assert isinstance(f, pr.classes.Field)

    assert f.fieldsize == (nvox, nsubj)
    assert f.D == 1
    assert f.masksize == (1, nvox)
    assert f.fibersize == nsubj
    assert f.field.shape == f.fieldsize
