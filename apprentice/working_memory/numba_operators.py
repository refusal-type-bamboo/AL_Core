from numbert.operator import BaseOperator
import math
from numba import njit, f8
import numpy as np
from .representation import numbalizer

textfield = {
    "id" : "string",
    "dom_class" : "string",
    # "offsetParent" : "string",
    "value" : "string",
    "contentEditable" : "number",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}

button = {
    "id": "string",
    "dom_class":"string",
    "label":"string",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}

checkbox = {
    "id": "string",
    "dom_class":"string",
    "label":"string",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
    "groupName":"string",  
}


component = {
    "id" : "string",
    "dom_class" : "string",
    # "offsetParent" : "string",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}

symbol = {
    "id" : "string",
    "value" : "string",
    "filled" : "number",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}

overlay_button = {
    "id" : "string",
}


numbalizer.register_specification("TextField",textfield)
numbalizer.register_specification("TextArea",textfield)
numbalizer.register_specification("Button", button)
numbalizer.register_specification("Checkbox", checkbox)
numbalizer.register_specification("RadioButton", checkbox)
numbalizer.register_specification("Component",component)
numbalizer.register_specification("Symbol",symbol)
numbalizer.register_specification("OverlayButton",overlay_button)


@njit(cache=True)
def is_prime(n):
    if n % 2 == 0 and n > 2:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


class Add(BaseOperator):
    commutes = True
    signature = 'float(float,float)'

    def forward(x, y):
        return x + y


class AddOne(BaseOperator):
    commutes = True
    signature = 'float(float)'

    def forward(x):
        return x + 1


class Subtract(BaseOperator):
    commutes = False
    signature = 'float(float,float)'

    def forward(x, y):
        return x - y


class Multiply(BaseOperator):
    commutes = True
    signature = 'float(float,float)'

    def forward(x, y):
        return x * y


class Divide(BaseOperator):
    commutes = False
    signature = 'float(float,float)'

    def condition(x, y):
        return y != 0

    def forward(x, y):
        return x / y


class Equals(BaseOperator):
    commutes = False
    signature = 'float(float,float)'

    def forward(x, y):
        return x == y


class Add3(BaseOperator):
    commutes = True
    signature = 'float(float,float,float)'

    def forward(x, y, z):
        return x + y + z


class Mod10(BaseOperator):
    commutes = True
    signature = 'float(float)'

    def forward(x):
        return x % 10


class Div10(BaseOperator):
    commutes = True
    signature = 'float(float)'

    def forward(x):
        return x // 10


class StrToFloat(BaseOperator):
    signature = 'float(string)'
    muted_exceptions = [ValueError]
    nopython = False

    def forward(x):
        return float(x)


class ReverseSign(BaseOperator):
    signature = 'float(float)'
    template = "ReverseSign(float)"
    nopython=False
    muted_exceptions = [ValueError]

    def forward(x):
        return -x


class VarName(BaseOperator):
    signature = 'str(str)'
    template = "VarName({})"
    nopython=False
    muted_exceptions = [ValueError]

    def forward(x):
        return x[x.find(next(filter(str.isalpha, x)))]


class FloatToStr(BaseOperator):
    signature = 'string(float)'
    template = 'FloatToStr({})'
    muted_exceptions = [ValueError]
    nopython = False

    def forward(x):
        if x == int(x):
            return str(int(x))
        return str(x)


class RipStrValue(BaseOperator):
    signature = 'string(TextField)'
    template = "RipStrValue({})"
    nopython=False
    muted_exceptions = [ValueError]

    def forward(x):
        return str(x.value)


class SkillSubtract(BaseOperator):
    signature = 'str(float)'
    template = "SkillSubtract({})"
    nopython=False
    muted_exceptions = [ValueError]

    def forward(x):
        if x == int(x):
            x = int(x)
        return 'subtract ' + str(x)


class SkillDivide(BaseOperator):
    signature = 'str(float)'
    template = "SkillDivide({})"
    nopython=False
    muted_exceptions = [ValueError]

    def forward(x):
        if x == int(x):
            x = int(x)
        return 'divide ' + str(x)


class GetCoefficient(BaseOperator):
    signature = 'str(str)'
    template = "GetCoefficient({})"
    nopython = False
    muted_exceptions = [ValueError]

    def condition(x):
        i = 0
        while i < len(x) and x[i] != 'x':
            i += 1
        return i < len(x)

    def forward(x):
        i = 0
        while i < len(x) and x[i] != 'x':
            i += 1
        j = i - 1
        while x[j].isdigit() or x[j] == '.':
            j -= 1
        if j == -1:
            j += 1
        if j == '-':
            j -= 1
        return x[j: i]


class GetBias(BaseOperator):
    signature = 'str(str)'
    template = "GetBias({})"
    nopython = False
    muted_exceptions = [ValueError]

    def condition(x):
        i = 0
        while i < len(x) and x[i] != 'x':
            i += 1
        return i < len(x)

    def forward(x):
        list = x.split('+')
        for l in list:
            if not ('x' in l or 'y' in l):
                i = 0
                while i < len(l) and (l[i] == '(' or l[i] == ' '):
                    i += 1
                l = l[i:]
                i = len(l) - 1
                while i >= 0 and (l[i] == ')' or l[i] == ' '):
                    i -= 1
                l = l[:i + 1]
                print('got %s' % l)
                return l