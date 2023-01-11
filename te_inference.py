from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network

def get_teinf(data_np, max_lag_sources = 1, min_lag_sources = 1):
    dt = Data(data_np, dim_order = "sp")
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': max_lag_sources,
                'min_lag_sources': min_lag_sources,
                'n_perm_max_stat': 500, # Maximum test
                'alpha_max_stat': 0.1,
                'n_perm_min_stat': 200, # Minimum test
                'alpha_min_stat': 0.1,
                'n_perm_omnibus': 500, # Omnibus
                'alpha_omnibus': 0.05,
                'permute_in_time': True}

    results = network_analysis.analyse_network(settings=settings, data=dt)

    return results

