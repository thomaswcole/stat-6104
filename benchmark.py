from nmf import *
import pandas as pd
import numpy as np
from collections import defaultdict
import timeit

def benchmark_faces(X_faces,methods,projection_types,r,runs):
    """ 
    Returns the benchmarked stats for the CBCL faces data
    """


    # --------------- Perform Benchmarking -------------------- # 
    stats = {
    'errors': {method: defaultdict(int) for method in methods},
    'time': {method: defaultdict(int) for method in methods},
    }

    # Set r
    r = 49
    runs = 10
    for method_name, method in methods.items():
        for projection in projection_types:
            total_errors = []    
            total_times = []   
            for i in range(runs):
                # Set seed per run
                seed = i + 1
                
                # Time NMF Method
                start_time = timeit.default_timer()
                _, _, errors = method(X_faces, r, random_state=seed,projection_type = projection)
                time = timeit.default_timer() - start_time

                # Store
                total_times.append(time)
                total_errors.append(errors[-1])
            
            # Store average times
            stats['time'][method_name][projection] = np.mean(total_times)
            stats['errors'][method_name][projection]= np.mean(total_errors)
            print(f"Completed {method_name},{projection}")

    # ----------------------- Clean Data -------------------- # 

    # Errors and Times
    errors_data = [
        (algo, proj, val) 
        for algo, projections in stats['errors'].items() 
        for proj, val in projections.items()
    ]

    time_data = [
        (algo, proj, val) 
        for algo, projections in stats['time'].items() 
        for proj, val in projections.items()
    ]

    # Create DataFrames
    errors_df = pd.DataFrame(errors_data, columns=['algorithm', 'projection', 'errors'])
    time_df = pd.DataFrame(time_data, columns=['algorithm', 'projection', 'time'])

    # Merge and reshape
    stats_df = (
        pd.merge(errors_df, time_df, on=['algorithm', 'projection'])
        .set_index(['algorithm', 'projection'])
        .unstack('projection')
    )
    return stats_df
