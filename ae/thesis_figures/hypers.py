import pandas as pd
from lightjob.cli import load_db

db = load_db()

def get_df():
    jobs = db.jobs_with()
    rows = []
    for j in jobs:
        col = {}
        if j['stats'] is None:
            continue
        for k, v in j['stats'].items():
            if type(v) is dict:
                for kk, v in v.items():
                    col[k+'_'+kk] = v
            else:
                col[k] = v
            
        try:
            col['stride'] = j['content']['train']['model']['params']['code_activations'][1]['params']['stride']
        except Exception:
            pass
        try:
            col['nb_layers'] = len(j['content']['train']['model']['params']['encode_nb_filters'])
        except Exception:
            col['nb_layers'] = len(j['content']['train']['model'][1]['params']['encode_nb_filters'])    

        try:
             col['zero_ratio'] = j['content']['train']['model']['params']['code_activations'][1]['params']['zero_ratio']
        except Exception:
            pass
        try:
            col['bottleneck'] = min(j['content']['train']['model']['params']['encode_nb_filters'])
        except Exception:
            pass
        try:
            col['noise'] = j['content']['train']['model'][0]['params']['params']['proba']
        except Exception:
            pass
        col['sampler'] = j['sampler']
        col['job_id'] = j['summary']
        #col['noise_count'] = 1.0 - (col['digits_count'] + col['letters_count'])
        #col['digits_object'] = 1 - col['digits_entropy']
        #col['letters_object'] = 1 - col['letters_entropy']
        #col['digits_and_letters_object'] = 1 - col['digits_and_letters_entropy']
        #col['emnist_object'] = 1 - col['emnist_letters_entropy']
        rows.append(col)
        
    df_full = pd.DataFrame(rows)
    df_full = df_full.set_index('job_id')
    return df_full
