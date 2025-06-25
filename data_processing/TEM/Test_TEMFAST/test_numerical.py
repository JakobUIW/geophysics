#%% Modules
from gp_tools.tem.survey_tem import SurveyTEM
import numpy as np

#%% Preprocess
survey = SurveyTEM('numerical/pelton/20250625')
survey.data_read()
survey.data_preprocess()

#%% First Look
# survey.plot_raw_filtered(filter_times=(7, 700), legend=True, fname='numerical_firstlook.png')



#%%
survey.plot_inversion(
        subset='N001',
        layer_type='linear',
        layers=1.5,
        max_depth=11,
        filter_times=(7, 700),
        start_model=np.ndarray((5,8),
                               buffer=np.array([50,50,50,50,50,50,50,50,#50,50,50,50,50,50,50,50,50,
                                                50,50,50,50,50,50,50,50,#50,50,50,50,50,50,50,50,50,
                                                0,0,0,0,0,0,0.5,0.5,#0.5,0.5,0,0,0,0,0,0,0,
                                                1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,5e-4,5e-4,#5e-4,5e-4,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,
                                                0.1,0.1,0.1,0.1,0.1,0.1,0.8,0.8,#0.8,0.8,0.1,0.1,0.1,0.1,0.1,0.1,0.1
                                                ])).transpose(),	
        lam=100,
        ip=True,
        verbose=True
    )



#%%