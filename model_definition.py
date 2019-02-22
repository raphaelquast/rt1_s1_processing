'''
definition of the model
'''
from rt1.volume import HenyeyGreenstein
from rt1.surface import LinCombSRF, HG_nadirnorm

# set the properties of the dataset
sig0 = True
dB = False

# define scattering distribution functions
def set_V_SRF(frac, omega, SM, VOD, v, v2, **kwargs):

    SRFchoices = [
        [frac, HG_nadirnorm(t=0.01, ncoefs=2, a=[-1., 1., 1.])],
        [(1. - frac), HG_nadirnorm(t=0.6, ncoefs=10, a=[1., 1., 1.])]
        ]
    SRF = LinCombSRF(SRFchoices=SRFchoices, NormBRDF=SM)

    V = HenyeyGreenstein(t=v, omega=omega, tau=v2 * VOD, ncoefs=8)
    return V, SRF

# specify the treatment of the parameters in the retrieval procedure
defdict = {
    'bsf'   : [False, 0.01, None,  ([0.01], [.25])],
    'v'     : [False, 0.4, None, ([0.01], [.4])],
    'v2'    : [True, 1., None, ([0.1], [1.5])],
    'VOD'   : [False, 'auxiliary'],
    'SM'    : [True, 0.1,  'D',   ([0.01], [0.2])],
    'frac'  : [True, 0.5, None,  ([0.01], [1.])],
    'omega' : [True, 0.3,  None,  ([0.05], [0.6])],
    }

#defdict['v'] = [True, .2, None, ([0.01], [.5])]
#defdict['v2'] = [False, 1.]
#defdict['VOD'] = [True, .25, None, ([0.01], [1.])]


# specify additional arguments for scipy.least_squares and rtfits.monofit
fitset = {'int_Q': False,
          '_fnevals_input': None,
          # 'verbosity' : 1, # verbosity of monofit
          # ------------ least_squares kwargs: ----------------
          'verbose': 0,  # verbosity of least_squares
          'ftol': 1.e-5,
          'gtol': 1.e-5,
          'xtol': 1.e-5,
          'max_nfev': 100,
          'method': 'trf',
          'tr_solver': 'lsmr',
          'x_scale': 'jac',
          }

