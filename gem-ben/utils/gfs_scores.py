
import pandas as pd

# COMPUTE GFS-scores
def get_gfs_score(res_file='real_results.h5')
    domains = ["Economic", "Biological", "Technological", "Social"]
    df_all = pd.read_hdf(res_file, 'df')
    df_all['LP MAP'] = df_all['LP MAP'].astype('float')
    df_all['LP P@100'] = df_all['LP P@100'].astype('float')
    micro_gfs_map = df_all.groupby("Method")['LP MAP'].mean()
    micro_gfs_map = micro_gfs_map/micro_gfs_map['rand']
    micro_gfs_p_100 = df_all.groupby("Method")['LP P@100'].mean()
    micro_gfs_p_100 = micro_gfs_p_100/micro_gfs_p_100['rand']
    print(' ############# Micro GFS MAP ############# ')
    print(micro_gfs_map)
    print(' ############# Micro GFS P@100 ############# ')     
    print(micro_gfs_p_100)

    macro_gfs_map = micro_gfs_map - micro_gfs_map
    macro_gfs_p_100 = micro_gfs_p_100 - micro_gfs_p_100
    for dom in domains:
        df_dom = df_all[df_all["Domain"] == dom]
        micro_gfs_map = df_dom.groupby("Method")['LP MAP'].mean()
        micro_gfs_map = micro_gfs_map/micro_gfs_map['rand']
        micro_gfs_p_100 = df_dom.groupby("Method")['LP P@100'].mean()
        micro_gfs_p_100 = micro_gfs_p_100/micro_gfs_p_100['rand']
        macro_gfs_map += micro_gfs_map
        macro_gfs_p_100 += micro_gfs_p_100
        print(' ############# Dom %s, Micro GFS MAP ############# ' % dom)
        print(micro_gfs_map)
        print(' ############# Dom %s, Micro GFS P@100 ############# ' % dom)     
        print(micro_gfs_p_100)
    print(' ############# Macro GFS MAP ############# ')
    print(macro_gfs_map)
    print(' ############# Macro GFS P@100 ############# ')
    print(macro_gfs_p_100)