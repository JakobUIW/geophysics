# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:42:19 2021

function library for 1D smooth pygimli inversion of tem data.
uses empymod as forward solver.

based upon pygimli example

@author: lukas aigner @ TU Wien, Research Unit Geophysics
"""

# %% import modules
import numpy as np
import pygimli as pg
from gp_tools.tem.forward import empymod_frwrd

# %% class_lib
class tem_smooth1D_fwd(pg.Modelling):
    """
    class for forward modeling of tem data with pyGIMLi and empymod
    """
    def __init__(self, depths, empy_frwrd, ip_modeltype_fwd=None):
        """Initialize the model and forward class from empymod."""   # adjustment where the depth vector is shortened
        self.dep = depths
        self.mesh1d = pg.meshtools.createMesh1D(len(self.dep))
        self.empy_frwrd = empy_frwrd
        self._nLayers = 0
        self._nPara = 4  # number of parameters per layer

        self._thk = None
        self._res = None
        self._dpt = None
        self.empy_frwrd = empy_frwrd
        self.ip_mdltype = 'pelton'
        self.layer_ip = 1
        self.return_rhoa = False
        self.resp_trafo = None
        super().__init__()
        self.setMesh(self.mesh1d)
        self.ip_modeltype_fwd = ip_modeltype_fwd

    def response(self, values):
        """Forward response of a given model."""
        print('empymod_inversion: tem_smooth1D_fwd: response called with values:{}\n'.format(values))
        print('empymod_inversion: tem_smooth1D_fwd: dep:{}\n'.format(self.dep[0:8]))
        print('empymod_inversion: tem_smooth1D_fwd: values shape:{}'.format(np.reshape(values, (8,4), 'C')))
        return self.empy_frwrd.calc_response(np.column_stack((self.dep[0:8], np.reshape(values, (8,4), 'C'))), ip_modeltype=self.ip_modeltype_fwd, 
                                             return_rhoa=self.return_rhoa, resp_trafo=self.resp_trafo)
        # return self.empy_frwrd.calc_response(np.column_stack((self.dep[0:12], values)), ip_modeltype=self.ip_modeltype_fwd)


class temip_block1D_fwd(pg.frameworks.Block1DModelling):
    """
    """
    def __init__(self, empy_frwrd, nPara=4, nLayers=4,
                 ip_mdltype='pelton', layer_ip=1, 
                 return_rhoa=False, resp_trafo=None, **kwargs):
        """

        Parameters
        ----------
        empy_frwrd : empymod_forwrd class
            DESCRIPTION.
        nPara : int, optional
            DESCRIPTION. The default is 1.
        nLayers : int, optional
            DESCRIPTION. The default is 4.
        ip_modeltype : string, optional
            DESCRIPTION. The default is 'pelton'.
        layer_ip : int, optional
            DESCRIPTION. The default is 1.
        return_rhoa : boolean, optional
            DESCRIPTION. The default is 1.
        resp_trafo : None or string, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._nLayers = 8
        self._nPara = nPara  # number of parameters per layer

        self._thk = None
        self._res = None
        self._dpt = None
        self.empy_frwrd = empy_frwrd
        self.ip_mdltype = ip_mdltype
        self.layer_ip = layer_ip
        self.return_rhoa = return_rhoa
        self.resp_trafo = resp_trafo

        # super(temip_block1D_fwd, self).__init__(**kwargs)
        super(pg.frameworks.Block1DModelling, self).__init__(**kwargs)
        # pg.frameworks.Block1DModelling.__init__(self, **kwargs)
        self._withMultiThread = True

        self.initModelSpace(nLayers)

    def response(self, model_vec):
        """Forward response."""

        return self.empy_frwrd.calc_response(model_vec,
                                             ip_modeltype=self.ip_mdltype,
                                             return_rhoa=self.return_rhoa,
                                             resp_trafo=self.resp_trafo)




class tem_inv_smooth1D(pg.Inversion):
    """
    derived class for the smooth inversion of TEM data
    """
    def __init__(self, setup_device):
        self.test_response = None
        self.start_model = None
        self.fwd = None
        self.nlayer = None
        self.max_depth = None
        self.depth_fixed = None
        self.setup_device = setup_device

        # super(pg.Inversion, self).__init__(**kwargs)
        super().__init__()


    def prepare_fwd(self, depth_vector, start_model, filter_times=None, max_depth=30, times_rx=None):
        """
        method to initialize the forward solver

        Parameters
        ----------
        times_rx : np.Array
            times for the receiver response.
        depth_vector : np.Array
            depth vector for the fixed layer inversion.
        start_model : np.Array
            resistivity vector describing the initial model.
        filter_times : tuple
            (t_min, t_may) in s, filtering range.
        max_depth : int, optional
            maximum depth of the fixed layer vector. The default is 30.

        Returns
        -------
        np.Array
            response of the start model.

        """
        self.max_depth = max_depth  # 4 * self.setup_device['tx_loop']
        self.start_model = start_model
        self.depth_fixed = depth_vector
        self.nlayer = len(self.depth_fixed)
        if len(self.depth_fixed) != len(self.start_model):
            raise ValueError('depth vector and start model have different lengths')
        
        empy_frwrd = empymod_frwrd(setup_device=self.setup_device,
                                   setup_solver=None,
                                   times_rx=times_rx,
                                   time_range=None, device='TEMfast',
                                   nlayer=self.nlayer, nparam=2)

        # self.depth_fixed = np.linspace(0., max_depth, self.nlayer)      #todo: fixed depth vector for inversion, add as parameter
        self.fop = tem_smooth1D_fwd(self.depth_fixed, empy_frwrd)
        # self.fop = temip_block1D_fwd(self.depth_fixed, empy_frwrd)


        self.test_response = self.fop.response(self.start_model)
        
        return self.test_response
    

    def prepare_fwd_ip(self, depth_vector, start_model, filter_times=None, max_depth=30, times_rx=None):
        """
        method to initialize the forward solver

        Parameters
        ----------
        times_rx : np.Array
            times for the receiver response.
        depth_vector : np.Array
            depth vector for the fixed layer inversion.
        start_model : np.Array
            ip array describing the initial model.
        filter_times : tuple
            (t_min, t_may) in s, filtering range.
        max_depth : int, optional
            maximum depth of the fixed layer vector. The default is 30.

        Returns
        -------
        np.Array
            response of the start model.

        """
        self.max_depth = max_depth  # 4 * self.setup_device['tx_loop']
        self.start_model = start_model
        self.depth_fixed = depth_vector
        self.nlayer = len(self.depth_fixed)
        print('tem_inv_smooth1D: prepare_fwd_ip: lengths of depth_fixed and start_model:')
        print(len(self.depth_fixed))
        print(len(self.start_model))
        if len(self.depth_fixed) != len(self.start_model):
            raise ValueError('depth vector and start model have different lengths')
        
        empy_frwrd = empymod_frwrd(setup_device=self.setup_device,
                                   setup_solver=None,
                                   times_rx=times_rx,
                                   time_range=None, device='TEMfast',
                                   nlayer=self.nlayer, nparam=start_model.shape[1])

        # self.depth_fixed = np.linspace(0., max_depth, self.nlayer)      #TODO: fixed depth vector for inversion, add as parameter
        self.depth_fixed_inv = np.stack((self.depth_fixed,) * 4, axis=0).flatten()
        print('tem_inv_smooth1D: prepare_fwd_ip: depth_fixed_inv shpe: {}'.format(self.depth_fixed_inv))
        # self.fop = temip_block1D_fwd(self.depth_fixed_inv, empy_frwrd)
        self.fop = tem_smooth1D_fwd(self.depth_fixed_inv, empy_frwrd)

        self.test_response = self.fop.response(self.start_model)
        self.parameterCount = self.start_model.shape[1]
        print('tem_inv_smooth1D: prepare_fwd_ip: parameterCount: {}'.format(self.parameterCount))
        
        return self.test_response


    def prepare_inv(self, maxIter=20, verbose=True):
        """
        Method to initialize the inversion

        Parameters
        ----------
        maxIter : int, optional
            maximum number of iterations. The default is 20.
        verbose : Boolean, optional
            show output of pygimli. The default is True.

        Returns
        -------
        None.

        """
        transRho = pg.trans.TransLogLU(1, 1000)
        transData = pg.trans.TransLog()

        self.verbose = verbose

        self.transModel = transRho
        self.transData = transData

        self.maxIter = maxIter

# %%
