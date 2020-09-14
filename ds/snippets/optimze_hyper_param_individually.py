#

import time

def time_spent(time0):
    t = time.time() - time0
    t_int = int(t) // 60
    t_min = t % 60
    if(t_int != 0):
        return '{}min {:.3f}s'.format(t_int, t_min)
    else:
        return '{:.3f}s'.format(t_min)

from sklearn.model_selection import GridSearchCV

def optimize_random_forest(mx_train, my_train, my_hyper_params, hyper_to_search, hyper_search_name, cv=4, scoring='accuracy'):
    """search best param to unic one hyper param.
    @mx_train, @my_train = x_train, y_train of dataset
    @my_hyper_params: dict with actuals best_params: start like: {}
      => will be accumulated and modified with each optimization iteration
      => example stater: best_hyper_params = {'random_state': 42, 'n_jobs': -1}
    @hyper_to_search: dict with key @hyper_search_name and list of values to gridSearch:
    @hyper_search_name: name of hyperparam
    """
    if(hyper_search_name in my_hyper_params.keys()):
        del my_hyper_params[hyper_search_name]
    if(hyper_search_name not in hyper_to_search.keys()):
        raise Exception('"hyper_to_search" dont have {} in dict'.format(hyper_search_name))
        
    t0 = time.time()
        
    rf = RandomForestClassifier(**my_hyper_params)
    
    grid_search = GridSearchCV(estimator = rf, param_grid = hyper_to_search, 
      scoring = scoring, n_jobs = -1, cv = cv)
    grid_search.fit(mx_train, my_train)
    
    print('took', time_spent(t0))
    
    data_frame_results = pd.DataFrame(
        data={'mean_fit_time': grid_search.cv_results_['mean_fit_time'],
        'mean_test_score_'+scoring: grid_search.cv_results_['mean_test_score'],
        'ranking': grid_search.cv_results_['rank_test_score']
         },
        index=grid_search.cv_results_['params']).sort_values(by='ranking')
    
    print('The Best HyperParam to "{}" is {} with {} in {}'.format(
        hyper_search_name, grid_search.best_params_[hyper_search_name], grid_search.best_score_, scoring))
    
    my_hyper_params[hyper_search_name] = grid_search.best_params_[hyper_search_name]
    
    """
    @@my_hyper_params: my_hyper_params appends best param find to @hyper_search_name
    @@data_frame_results: dataframe with statistics of gridsearch: time, score and ranking
    @@grid_search: grid serach object if it's necessary
    """
    return my_hyper_params, data_frame_results, grid_search

"""
# EXAMPLE in SPAM Detector

best_hyper_params = {'random_state': 42, 'n_jobs': -1} # Stater Hyper Params
search_hyper = { 'criterion' :['gini', 'entropy'] }

best_hyper_params, results, last_grid_search = optimize_random_forest(
    X_train_dtm, y_train, best_hyper_params, search_hyper, 'criterion')
"""
