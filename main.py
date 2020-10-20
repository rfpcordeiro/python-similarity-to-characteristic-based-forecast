import seaborn as sns
import statistics
import numpy as np
import time
import pandas as pd
import constants as c
import os
import concurrent.futures
from tqdm import tqdm
from connectors import connector_gbq
from sklearn.preprocessing import StandardScaler
  
def export_similarity_file(df_new, df_compared, export_file_name):
    """
    Export result of the similarity analysis to a .csv
    
    Parameters
    ----------
    df_new : pandas.DataFrame
        data frame object with new products data
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
    export_file_name : str
        name wanted to the file which is going to be exported
        
    Returns
    -------
    None
    """
    df_new = df_new.reset_index()
    df_compared = df_compared.reset_index()
    # define a list with the from where you get the new location-product code, 
    # the similar location-product code that you've found and the similarity factor
    # and rename it to the needed format as that software requires
    lst_result = [
        df_new[['PLNT_CD', 'MTRL_CD']].rename(columns={'PLNT_CD':'location','MTRL_CD':'product'}),
        df_compared[['PLNT_CD', 'MTRL_CD', 'scaling_factor']].rename(
            columns={'PLNT_CD':'reference_location_code','MTRL_CD':'reference_code'}),]
    # make this list become a dataframe
    df_export = pd.concat(lst_result, axis=1, sort=False)
    # correct the data type and the insert leading zeros
    for col in ['location', 'product', 'reference_location_code', 'reference_code']:
        df_export[col] = df_export[col].astype(int)
        df_export[col] = df_export[col].astype(str)
    for col in ['location', 'reference_location_code']:
        df_export[col]=df_export[col].apply(lambda x: x.zfill(4))
    # append new rows to the file
    with open(export_file_name,'a') as fd:
        df_export.to_csv(fd, sep=';', decimal='.', index=False, header=False)
    
def read_data(new_data_file_name, download_ind):
    """
    Read data from input file and donwload old data from the database
    
    Parameters
    ----------
    new_data_file_name : str
        full path to the file that stores the data of new products
    download_ind :
        indicator that shows if is necessary to update the old product data base
    Returns
    -------
    df_old : pandas.DataFrame
        data frame object with old products data
    df_new : pandas.DataFrame
        data frame object with new products data
    """
    if download_ind:
        # update old products data base
        connector = connector_gbq.ConnectorGBQ()
        df_old = connector.read_query(query=c.query)
        # save the update to a local file to speed up any new analysis
        df_old.to_csv(get_input_file_path("samples.csv"), sep=';', decimal='.', index=False)
    else:
        # if isn't necessary to update, read from file
        df_old = pd.read_csv(get_input_file_path("samples.csv"), sep=';')
    print(f'{time.ctime()}, Old materials data read')
    # read new products data
    df_new = pd.read_excel(get_input_file_path(new_data_file_name))
    print(f'{time.ctime()}, New materials data read')
    # rename columns
    df_new = df_new.rename(columns=c.dict_translate_new)
    return df_old, df_new
    
def get_input_file_path(file_name):
    """
    Set the path in input folder to a file
    
    Parameters
    ----------
    file_name : str
        file that you want to set at input folder 
    Returns
    -------
    full_out_path : str
        full input file path with file name and format
    """
    # define the folder we desired to be the input folder
    out_dir = os.path.join(os.path.abspath(""),'data')
    # check if the folder exists
    if os.path.isdir(out_dir):
        return os.path.join(os.path.abspath(""), 'data', file_name)
    else:
        return os.path.join(os.path.abspath(""), file_name)

def get_output_file_path(file_name):
    """
    Set the path in output folder to a file
    
    Parameters
    ----------
    file_name : str
        file that you want to set at output folder 
    Returns
    -------
    full_out_path : str
        full output file path with file name and format
    """
    # define the folder we desired to be the output/result folder
    out_dir = os.path.join(os.path.abspath(""),'result')
    # check if the folder exists
    if not os.path.isdir(out_dir):
        # if not, create it
        os.mkdir(out_dir)
    # return the full path (directory + file name + file extension)
    full_out_path = os.path.join(out_dir, file_name)
    return full_out_path

def one_hot_encoder(df, col_nm):
    """
    Aplly one hot enconding in a column of a data frame
    
    Parameters
    ----------
    df : pandas.DataFrame
        data frame you want to apply the OHE
    col_nm : str
        column name you want to apply the OHE
    Returns
    -------
    df_res : pandas.DataFrame
        data frame after applying OHE
    """
    # create a list with the original data frame and another dataframe with the 
    # columns created by yhe OHE
    lst_result = [df, pd.get_dummies(df[col_nm], prefix=col_nm)]
    # concatenate both data frames
    df_res = pd.concat(lst_result, axis=1, sort=False)
    # delete the columns used in OHE
    del df_res[col_nm]
    return df_res

def get_smallest_distance(lst):
    """
    Calculate the smallest distance of the numeric characterist of old materials and the new one
    
    Parameters
    ----------
    lst : list
        list with two elements.
        lst[0] : one row of the df_new
        lst[1] : whole df_old data frame
        
    Returns
    -------
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
    """
    df_result = pd.DataFrame()
    new_obs = lst[0]
    df_old = lst[1]
    # configure the new register as an array
    new_obs_arr = np.array(new_obs)
    # calculate the euclidean distance between the new register and each old one
    idx_closest = np.argmin([np.linalg.norm(new_obs_arr - np.array(x)) for x in df_old.values])
    # with the index of the most similar row, selext it on the data frame how the result
    # of iloc is a Series object convert it to a data frame again the result data frame is
    # transposed, so we need to transpose it again to original format at end reset it's index 
    df_aux = df_old.iloc[idx_closest].to_frame().transpose().reset_index()
    # check if the result data frame exists
    if len(df_result)==0:
        # if don't, create it
        df_result = df_aux.copy()
    else:
        # if it already exists, append data to it
        df_result = df_result.append(df_aux)
    return df_result

def scale_data(df_new, df_old):
    """
    Apply Standard Scaling to the old and new products
    
    Parameters
    ----------
    df_old : pandas.DataFrame
        data frame object with old products data
    df_new : pandas.DataFrame
        data frame object with new products data
        
    Returns
    -------
    df_new_scaled : pandas.DataFrame
        data frame object with new products data after standard scaler process
    df_old_scaled : pandas.DataFrame
        data frame object with old products data after standard scaler process
    """
    # create a new data frame concatenating the old and the new products data
    # this way the data is scaled considering all rows and not only each one separately
    df = pd.concat([df_new, df_old])
    # instanciate the standard scaler object
    scaler = StandardScaler()
    # fit the scaler
    scaler.fit(df[c.cols_num])
    # apply scaling to the data frame
    scaled_features = scaler.transform(df[c.cols_num])
    # create a new data frame with the scaled data and get columns data from the old data frame
    df_feat = pd.DataFrame(scaled_features, columns=c.cols_num)
    # create a column with the index of original data frame
    df_feat[df.index.name] = df.index
    # define a list with the categorical columns
    cols_cat = [col for col in df.columns if col not in c.cols_num]
    # copy index from original to the scaled one
    df_feat = df_feat.set_index(df.index.name)
    # copy the categorical columns from the original data frame
    for col in cols_cat:
        df_feat[col] = df[col]
    # how the data is concatenated we need to separate them again
    # get the length from each original data frame
    old_len_new = len(df_new)
    old_len_old = len(df_old)
    # split them based on their length
    df_new_scaled = df_feat[:old_len_new]
    df_old_scaled = df_feat[old_len_new:]
    return df_new_scaled, df_old_scaled

def reverse_one_hot_encoder(df, col_nm):
    """
    Reverse the one hot encoding process
    
    Parameters
    ----------
    df : pandas.DataFrame
        data frame object with the old products that are more similar the new products
    col_nm : str
        name of the original column
        
    Returns
    -------
        df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products without the OHE procress
    """
    # create a list with all columns names that has in name the original name
    lst_stack = [x for x in df.columns if x.find(col_nm)>=0]
    # select only this columns and save it in a auxiliar data frame
    df_aux = df[lst_stack]
    # concatenate the result data frame with the column reversed one hot encoder 
    df[col_nm] = df_aux[df_aux==1].stack().reset_index().drop(0,1)['level_1']\
            .apply(lambda x: int(x.replace(col_nm+'_',''))).to_list()
    # and delete the columns generated by the one hot encoding process
    for col in lst_stack:
        df = df.drop(col, axis=1)
    return df

def calculate_scaling_factor(df_new, df_result, df_new_orig_prc, df_old_orig_prc, ind_price_rt = False):
    """
    Calculate the scaling factor that should be used in your forecast system/algorithm to set how much you should copy from the old product's forecast to the new one
    
    Parameters
    ----------
    df_new : pandas.DataFrame
        data frame object with new products data
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
        
    Returns
    -------
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products, adding a column saying the scaling factor between them
    """
    lst_dist = []
    lst_sf = []
    lst_new_prc = []
    lst_old_prc = []
    # for each row of the data frame
    for i in range(len(df_new)):
        # calculate again the euclidean distance between the new record and the most similar one
        # how df_result has a index column that is the index of df_old it has one more column than df_new
        # to make them equal we need to set how columns we want to use from df_result
        diff_arr = df_new.values[i] - df_result.values[i]
        euclidean_distance = np.linalg.norm(diff_arr)
        lst_dist.append(euclidean_distance)
        lst_new_prc.append(
            df_new_orig_prc[
                (df_new_orig_prc['PLNT_CD']==df_new.iloc[i]['PLNT_CD'])&
                (df_new_orig_prc['MTRL_CD']==df_new.iloc[i].name)]['PRC_SLS_VKP0'].values)
        lst_old_prc.append(
            df_old_orig_prc[
                (df_old_orig_prc['PLNT_CD']==df_result.iloc[i]['PLNT_CD'])&
                (df_old_orig_prc['MTRL_CD']==df_result.iloc[i].name)]['PRC_SLS_VKP0'].values)
    # how we need to see the % of difference between the 2 arrays and there is no obvious way to calculate it
    for i in range(len(lst_dist)):
        euclidean_distance = lst_dist[i]
        if euclidean_distance == 0:
            # 0 distance means that there is a product in old data frame that has exactly the same
            # characteristics than the new one
            scaling_factor = 1.00 
        else:
            # if it is not the case apply the "trick" which is a mathematical function that decreases 
            # as the distance increases and is smoothed by the mean value of all distances
            scaling_factor = np.exp(-(euclidean_distance**2/statistics.mean(lst_dist)**2))
        if ind_price_rt:
            if lst_new_prc[i] > lst_old_prc[i]:
                scaling_factor = (lst_old_prc[i] / lst_new_prc[i]) * scaling_factor
            else:
                scaling_factor = (lst_new_prc[i] / lst_old_prc[i]) * scaling_factor
        scaling_factor = (lst_old_prc[i] / lst_new_prc[i]) * scaling_factor
        scaling_factor = round(float(scaling_factor), 2)
        if scaling_factor<0.25:
            scaling_factor = 0.25
        if scaling_factor>4:
            scaling_factor = 4
        # save the result on a list
        lst_sf.append(scaling_factor)
    # get the list with all scaling factors and merge it with df_result as a new column
    df_result['scaling_factor'] = lst_sf
    return df_result

def get_columns_createded_by_ohe(df, lst_ohe): 
    """
    Get a list with the names of the columns that were created by the one hot encoding process
    
    Parameters
    ----------
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
    lst_ohe : list
        list with the orignal name of the columns that have passed through one hot enconding process
        
    Returns
    -------
        flat_list : list with the name of the columns after them have passed through one hot enconding process
    """
    lst_one_hot_encoded_cols=[]
    flat_list = []
    for col_nm in lst_ohe:
        # create a list with all columns names that has in name the original name
        lst_one_hot_encoded_cols.append([x for x in df.columns if x.find(col_nm)>=0])
    # convert it from a list of lists into a list with all elements
    for sublist in lst_one_hot_encoded_cols:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def reverse_scaler(df_result, df_old, lst_cols_created_ohe):
    """
    Reverse the Standard Scaler process
    
    Parameters
    ----------
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
    df_old : pandas.DataFrame
        data frame object with old products data
    lst_cols_created_ohe : list
        list with the names of the columns that were created by the one hot encoding process
        
    Returns
    -------
    df_result_wthout_ohe : pandas.DataFrame
        data frame with the standard scaler process reversed
    """
    # remove the columns that were created by the one hot encoding process
    df_result_wthout_ohe = df_result.drop(labels=lst_cols_created_ohe, axis=1)
    # locate the rows of old data frame that were considered the better combination for each new product
    # looking only the columns that were created by OHE
    df_aux = df_old.loc[df_result.index][lst_cols_created_ohe]
    for col in df_aux.columns:
        df_result_wthout_ohe[col] = df_aux[col]
    return df_result_wthout_ohe

def get_smallest_distance_multiprocessing(df_new,df_old):
    """
    Calculate the smallest distance of the numeric characterist of old materials and the new one, using multi-processing
    
    Parameters
    ----------
    df_new : pandas.DataFrame
        data frame object with new products data
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
        
    Returns
    -------
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
    """
    lst_rows = []
    exe_ind = False
    c_n = c.core_numbers
    # to use multi-processing we need to define a function that requires only one parameter
    # and how we need to compare two things we create a list where we pass these 2 elements
    # which we want to compare as a row of the list
    for row in df_new.values:
        # each row of the list is composed by the new material and all old materials that
        # we could copare with
        lst_rows.append([row,df_old])
    
    # how we want to speed up and we can actually try to use de maximum availiable cores
    # to do the multi-processing and if it is not possible we retry it with subtracting one core
    # and while it has not finished we try again
    while exe_ind==False:
        try:
            # set multi-processing
            with concurrent.futures.ProcessPoolExecutor(c_n) as executer:
                # apply desired function (this case is to get the smallest distance)
                lst_results = list(tqdm(executer.map(get_smallest_distance, lst_rows),  total=len(lst_rows)))
            # at end make the result become a data frame
            df_result = pd.concat(lst_results, axis=0)
            # change de indicator to not repeat the process unecessary
            exe_ind=True
        except:
            # once a problem with the cores has been shown
            if c_n == 1:
                # if the core is one try to "recharge" it
                c_n = c.core_numbers
            else:
                # if it not the minial yet, subtract one of it
                c_n -= c_n
    df_result = df_result.rename(columns={'index':df_new.index.name}).set_index(df_new.index.name)
    return df_result

def get_similar_rows(df_old, df_new, lst_ohe, df_old_orig_prc, df_new_orig_prc):
    """
    Create a data frame object with similar old products to the new ones
    
    Parameters
    ----------
    df_old : pandas.DataFrame
        data frame object with old products data
    df_new : pandas.DataFrame
        data frame object with new products data
    lst_ohe : list
        list with the name of the columns that exist in both data frames and are going to pass through one hot enconding process
    Returns
    -------
    df_result : pandas.DataFrame
        data frame object with the old products that are more similar the new products
    """
    df_new = df_new.set_index('MTRL_CD')
    df_old = df_old.set_index('MTRL_CD')
    # set initial variables that are going to be used in the process
    df_aux = pd.DataFrame()
    lst_orig_col = df_old.columns
    # apply one hot encoding to the new and old data frames
    for col in lst_ohe:
        df_old = one_hot_encoder(df=df_old, col_nm=col)
        df_new = one_hot_encoder(df=df_new, col_nm=col)   
    # how the data frame with new records may not have all possible registers
    # we must creat a list with all columns that don't exist on it and fill
    # them with 0
    for item in [col for col in df_old.columns if col not in df_new.columns]:
        df_new[item] = 0
    # reorder the data frame so both follows the same sequence
    df_new = df_new[df_old.columns]
    # apply Standard Scaler to the data in data frames
    df_new_scaled, df_old_scaled = scale_data(df_new, df_old)  
    for ele in [df_new_scaled, df_old_scaled]:
        ele['PRC_SLS_VKP0'] *= 5
    # calculate the euclidean distance between the products and keep the smallest
    df_result = get_smallest_distance_multiprocessing(df_new=df_new_scaled, df_old=df_old_scaled)
    # after it we need to revert the process of one hot encoding to get back the values as unique column
    # with it know from wich plant the product came from
    for col_nm in lst_ohe:
        df_result = reverse_one_hot_encoder(df = df_result, col_nm=col_nm)  
        df_new_scaled = reverse_one_hot_encoder(df = df_new_scaled, col_nm=col_nm)  
        df_old_scaled = reverse_one_hot_encoder(df = df_old_scaled, col_nm=col_nm) 
    # calculate the scaling factor of the new products
    df_result = calculate_scaling_factor(df_new_scaled, df_result, df_new_orig_prc, df_old_orig_prc)
    return df_result
 
def execute_grouped(input_file_name, output_file_name, plnt_col_nm, grp_col_nm, update_base_ind = True):
    """
    Create a file with similarity comparation between new products and the products that already exists in your database
    
    Parameters
    ----------
    input_file_name : str
        name of the file used as input that has the new products codes and characteristics
    output_file_name : str
        name of the file created after the code run
    plnt_col_nm : str
        name of the column that has the location/plant code or name
    grp_col_nm : str
        name of the column that has the group code or name
    update_base_ind : boolean
        indicator that shows if is necessary to update the old products data base or could be used the file that we already have downloaded
        
    Returns
    -------
    None
    """
    # how we use multi-rocessing to speed up the code we need to create an empty csv that will
    # store the result of the analysis
    # this csv must have only the header, so this way we can append only the new lines to it
    with open(get_output_file_path(output_file_name),'w') as file:
        # define the csv header
        file.write('location;product;reference_location_code;reference_code;scaling_factor')
        file.write('\n')
    # read the new and old products data
    df_old, df_new = read_data(new_data_file_name=input_file_name, download_ind=update_base_ind)
    # for each plant code in new products base
    for plnt in df_new[plnt_col_nm].unique():
        # for each group code that exists in this plant in the new products base
        for grp in df_new[df_new[plnt_col_nm]==plnt][grp_col_nm].unique():
            print(f'{time.ctime()}, Start process for Plant {plnt} and Group {grp}')
            # define two new auxiliar dataframes, they represent the new and old products that are
            # going to be used in the analysis
            df_new_aux = df_new[(df_new[plnt_col_nm]==plnt)&(df_new[grp_col_nm]==grp)].reset_index(drop=True)
            df_old_aux = df_old[(df_old[plnt_col_nm]==plnt)&(df_old[grp_col_nm]==grp)].reset_index(drop=True)
            df_new_orig_prc = df_new_aux[['PLNT_CD','MTRL_CD', 'PRC_SLS_VKP0']]
            df_old_orig_prc = df_old_aux[['PLNT_CD','MTRL_CD', 'PRC_SLS_VKP0']]
            # one thing we need to check is if exists old products to compare, once we have some rules to 
            # considere before adding a product to the list of old products
            if len(df_old_aux):
                # define similarity
                df_result = get_similar_rows(
                    df_old_aux, df_new_aux, [plnt_col_nm,grp_col_nm], df_old_orig_prc, df_new_orig_prc)
                # export file
                export_similarity_file(
                    df_new = df_new_aux, 
                    df_compared = df_result, 
                    export_file_name = get_output_file_path(output_file_name))
            else:
                # if there is no product to compare we just jump this group code
                pass