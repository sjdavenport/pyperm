"""
    Random field classes
"""
import numpy as np
import pyperm as pr

class Field:
    """ Field class

    Parameters
    ----------
    field:  a numpy.ndarray of shape (Dim) or (Dim, fibersize)
      Here Dim is the size of the field and fibersize is an index for the fields,
      typically fibersize is the number of subjects.

    mask:  Bool,
      a boolean numpty array giving the spatial mask the size of which
           must be compatible with the field

    Returns
    -------
    An object of class field

    Examples
    --------
    # 1D
    field = np.random.randn(100, 30)
    mask = np.ones((100, 1), dtype=bool)
    exField = pr.Field(field, mask)
    print(exField)
    
    # 2D
    field = np.random.randn(100, 100, 30)
    mask = np.ones((100, 100), dtype=bool)
    exField = pr.Field(field, mask)
    print(exField)
    
    # 2D no subjects
    field = np.random.randn(100, 100)
    mask = np.ones((100, 100), dtype=bool)
    exField = pr.Field(field, mask)
    print(exField)
    -----------------------------------------------------------------------------
    """
    def __init__(self, field, mask):
        self.field = field
        self.fieldsize = field.shape
        masksize = mask.shape

        # Check that the mask is a boolean array
        if mask.dtype != np.bool:
            raise Exception("The mask must be a boolean array")

        # Assign the dimension
        self.D = len(masksize)

        # Cover the 1D case where the mask is a vector!
        # (Allows for row and column vectors)
        if (self.D == 2) and (masksize[0] == 1 or masksize[1] == 1):
            self.D = 1
            # Force the mask to be a row vector
            if masksize[1] == 1:
                mask = mask.transpose()
            self.masksize = tuple(np.sort(masksize))
        else:
            # In D > 1 just assign the mask size
            self.masksize = masksize

        # Obtain the fibersize
        if self.masksize == self.fieldsize:
            # If the size of the mask is the size of the data then there is just
            # one field so the fibersize is set to 1
            self.fibersize = 1
        else:
            self.fibersize = self.field.shape[self.D:]
            if len(self.fibersize) == 1:
                self.fibersize = self.field.shape[self.D:][0]
            elif self.masksize == (1, 1):
                self.fibersize = self.field.shape[self.D + 1:][0]

        # Ensure that the size of the mask matches the size of the field
        if self.D > 1 and field.shape[0: self.D] != self.masksize:
            raise Exception("The size of the spatial field must match the mask")
        elif self.D == 1 and field.shape[0: self.D][0] != self.masksize[1]:
            # If the size of the mask doesn't match the field then return an error
            raise Exception("The size of the spatial field must match the mask")

        # If it passes the above tests assign the mask to the array
        self.mask = mask

    def __str__(self):
        # Initialize string output
        str_output = ''

        # Get a list of the attributes
        attributes = vars(self).keys()

        # Add each attribute (and its properties to the output)
        for atr in attributes:
            if atr in ['D', 'fibersize']:
                str_output += atr + ': ' + str(getattr(self, atr)) + '\n'
            elif atr in ['_Field__mask']:
                pass
            elif atr in ['_Field__fieldsize']:
                str_output += 'fieldsize' + ': ' + str(getattr(self, atr)) + '\n'
            elif atr in ['_Field__masksize']:
                str_output += 'masksize' + ': ' + str(getattr(self, atr)) + '\n'
            elif atr in ['_Field__field']:
                str_output += 'field' + ': ' + str(getattr(self, atr).shape) + '\n'
            else:
                str_output += atr + ': ' + str(getattr(self, atr).shape) + '\n'

        # Return the string (minus the last \n)
        return str_output[:-1]

    #Getting and setting field
    def _get_field(self):
        return self.__field

    def _set_field(self, value):
        if hasattr(self, 'mask'):
            if self.D > 1:
                if value.shape[0:self.D] != self.masksize:
                    raise ValueError("The size of the field must be compatible with the mask")
            else:
                if value.shape[0:self.D][0] != self.masksize[1]:
                    raise ValueError("The size of the field must be compatible with the mask")
        self.__field = value
        self.fieldsize = value.shape

    #Getting and setting mask
    def _get_mask(self):
        return self.__mask

    def _set_mask(self, value):
        if (self.D > 1) and value.shape != self.masksize:
            raise ValueError("The size of the mask must be compatible with the field")
        elif (self.D == 1) and tuple(np.sort(value.shape)) != self.masksize:
            raise ValueError("The size of the mask must be compatible with the field")
        if  value.dtype != np.bool:
            raise Exception("The mask must be a boolean array")
        self.__mask = value
        self.masksize = value.shape

    #Getting and setting fieldsize
    def _get_fieldsize(self):
        return self.__fieldsize

    def _set_fieldsize(self, value):
        if value != self.field.shape:
            raise Exception("The field size cannot be changed directly")
        self.__fieldsize = value

    #Getting and setting masksize
    def _get_masksize(self):
        return self.__masksize

    def _set_masksize(self, value):
        if hasattr(self, 'mask'):
            if value != self.mask.shape:
                raise Exception("The field size cannot be changed directly")
        self.__masksize = value

    # Set properties
    field = property(_get_field, _set_field)
    mask = property(_get_mask, _set_mask)
    fieldsize = property(_get_fieldsize, _set_fieldsize)
    masksize = property(_get_masksize, _set_masksize)

    
def make_field(array, fibersize=1):
    """ conv2field converts a numpy array to am object of class field

    Parameters
    ----------
    array:  numpy.ndarray of shape (Dim, fibersize),
            Here Dim is the spatial size and fibersize is the index dimension
    fibersize: int,
               specifies the size of the fiber, typically this is 1 i.e. when the
               last dimension of array corresponds to the fibersize

    Returns
    -------
    F: object of class field

    Examples
    --------
    data = np.random.randn(100, 30)
    F = pr.make_field(data)
    """
    fieldsize = array.shape
    D = len(fieldsize) - fibersize
    if D == 1:
        masksize = (fieldsize[0], 1)
    else:
        masksize = fieldsize[0:D]
    mask = np.ones(masksize, dtype = bool)

    f = pr.Field(array, mask)
    return f
