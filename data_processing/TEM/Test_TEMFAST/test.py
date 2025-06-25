#%% Modules
from gp_tools.tem.survey_tem import SurveyTEM
import numpy as np

#%% Preprocess

survey = SurveyTEM('donauinsel/20250419')
survey.data_read()
survey.data_preprocess()

#%% First Look
soundings_15 = [i for i in range(1, 15)]
sounding_6 = [16, 25, 26, 27]

name_soundings_15 = ['D'+str(i).zfill(3) for i in soundings_15]
name_soundings_6 = ['D'+str(i).zfill(3) for i in sounding_6]
name_soundings = name_soundings_15 + name_soundings_6
# #%%
# survey.plot_raw_filtered(filter_times=(7, 700), legend=True, subset=name_soundings_15, fname='15m_firstlook.png')


#%%
survey.plot_inversion(
        subset='D001',
        layer_type='dict',
        layers={0:2, 9:5, 30:10},
        max_depth=51,
        filter_times=(10, 100),
        start_model=np.ndarray((5,12),
                               buffer=np.array([50,50,50,50,50,50,50,50,50,50,50,50,
                                            50,50,50,50,50,50,50,50,50,50,50,50,
                                            0,0,0,0,0,0,0.5,0.5,0.5,0.5,0,0,
                                            1e6,1e6, 1e6, 1e6, 1e6, 1e6, 5e4, 5e4, 5e4, 5e4, 1e6, 1e6,
                                            0.1,0.1,0.1,0.1,0.1,0.1,0.8,0.8,0.8,0.8,0.1,0.1])).transpose(),	
        lam=100,
        ip=True,
        verbose=True
    )



#%%



# %% L-curve

for s in name_soundings_15:
    survey.l_curve_plot(
        sounding=s,
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=50,
        test_range=(10, 1000, 20),
        filter_times=(10, 700)
    )

for s in name_soundings_6:
    survey.l_curve_plot(
        sounding=s,
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=20,
        test_range=(10, 1000, 20),
        filter_times=(10, 100)
    )

#%%
for s in name_soundings_15:
    survey.lambda_analysis_comparison(
        sounding=s,
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=50,
        test_range=(10, 1000, 20),
        filter_times=(10, 700)
        )
    
for s in name_soundings_6:
    survey.lambda_analysis_comparison(
        sounding=s,
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=20,
        test_range=(10, 1000, 20),
        filter_times=(10, 100)
        )

#%%
for s in name_soundings_15:
    survey.optimised_inversion_plot(
        sounding=s,
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=50,
        test_range=(10, 1000, 20),
        filter_times=(10, 700),
        lam=140
        )
    
for s in name_soundings_6:
    survey.optimised_inversion_plot(
        sounding=s,
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=20,
        test_range=(10, 1000, 20),
        filter_times=(10, 100),
        lam=140
        )


#%%
for s in name_soundings_15:
    survey.plot_inversion(
        subset=[s],
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=50,
        filter_times=(10, 700),
        lam=55,
    )

for s in name_soundings_6:
    survey.plot_inversion(
        subset=[s],
        layer_type='dict',
        layers={0:0.5, 5:1, 15:2},
        max_depth=20,
        filter_times=(10, 100),
        lam=55,
    )
# %%
