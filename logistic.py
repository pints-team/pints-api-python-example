from __future__ import print_function
import pints
import pints.toy as toy
import numpy as np
import matplotlib.pyplot as pl
import requests
import copy


class LogisticAPI(pints.LogPDF):
    def __init__(self, base_url):
        self.base_url =  base_url
        r = requests.get(base_url + 'logistic-model/n_parameters')
        self.n = int(r.text)

    def __call__(self, x):
        if len(x) != self.n:
            raise ValueError('incorrect number of parameters')
        payload = {'x': [str(i) for i in x]}
        r = requests.get(self.base_url + 'logistic-model', params=payload)
        return float(r.text)

    def evaluateS1(self, x):
        if len(x) != self.n:
            raise ValueError('incorrect number of parameters')
        payload = {'x': [str(i) for i in x]}
        r = requests.get(self.base_url + 'logistic-model/evaluateS1', params=payload)
        return_array = r.json()
        L = float(return_array[0])
        L_dash = np.array(return_array[1:], dtype=float)
        return (L, L_dash)

    def n_parameters(self):
        return self.n

lpdf = LogisticAPI('https://mighty-badlands-12664.herokuapp.com/pints-team/benchmarks/1.0.0/')

real_parameters = [0.015, 500, 10]

## Select some boundaries
boundaries = pints.RectangularBoundaries([0, 400, 0], [0.03, 600, 20])

## Perform an optimization with boundaries and hints
x0 = 0.01, 450, 5
sigma0 = [0.01, 100, 10]
found_parameters, found_value = pints.optimise(
    lpdf,
    x0,
    sigma0,
    boundaries,
    method=pints.CMAES
    )

## Show score of true solution
print('log likelihood at true solution: ')
print(lpdf(real_parameters))
#
## Compare parameters with original
print('Found solution:          True parameters:' )
for k, x in enumerate(found_parameters):
    print(pints.strfloat(x) + '    ' + pints.strfloat(real_parameters[k]))


