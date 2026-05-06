import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from math import ceil, floor
import pickle as pkl

def preprocess_foursquare_data(root_path, city='NYC', scale=False, size=(40, 30, 10), interval=None):
    """
    Preprocesses the Foursquare dataset and saves it as an .npz file.

    Args:
        root_path (str): Root directory path where the dataset is located.
        city (str): City name, either 'NYC' or 'TKY'.
        scale (bool): Whether to scale the numerical features.
        output_filename (str): Name of the output .npz file.
    """
    # Define the data path
    data_path = os.path.join(root_path, f"dataset/foursquare/foursquare_{city}.csv")
    print(f"Loading data from {data_path}")

    # Read the CSV file
    dataframe = pd.read_csv(data_path)

    # Convert 'Timestamp' to datetime
    dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])

    # Extract temporal features
    dataframe['Hour'] = dataframe['Timestamp'].dt.hour
    dataframe['DayOfWeek'] = dataframe['Timestamp'].dt.dayofweek
    dataframe['Day'] = dataframe['Timestamp'].dt.day
    dataframe['Month'] = dataframe['Timestamp'].dt.month

    # Select relevant features
    feature_cols_input = ['User ID', 'Hour', 'DayOfWeek', 'Day', 'Month', 'Latitude', 'Longitude', 'Venue ID']
    feature_cols_output = ['User ID', 'Hour', 'DayOfWeek', 'Day', 'Month']
    label_col_output = 'Venue ID'

    # Initialize scaler if scaling is required
    if scale:
        scaler = StandardScaler()
        dataframe[feature_cols_input] = scaler.fit_transform(dataframe[feature_cols_input])
        print("Numerical features scaled.")

    # Initialize lists to store sequences
    POI_categories = []
    input_seq_feature = []
    predict_seq_feature = []
    predict_seq_label = []
    timestamps = []

    # Unique user IDs
    uid_set = dataframe['User ID'].unique()
    print(f"Total unique users: {len(uid_set)}")

    # Iterate over each user to generate sequences
    for uid in uid_set:
        user_data = dataframe[dataframe['User ID'] == uid]
        user_data = user_data.sort_values(by='Timestamp').reset_index(drop=True)

        # Extract relevant columns
        input_features = user_data[feature_cols_input].values  
        output_features = user_data[feature_cols_output].values  
        output_labels = user_data[label_col_output].values  # [VenueID]
        timestamps_user = user_data['Timestamp'].values  # numpy array of Timestamps
        
        seq_len = size[0]
        pred_len = size[2]
        if interval is None:
            interval = seq_len + pred_len  # Default interval

        num_sequences = (len(user_data) - seq_len - pred_len + 1) // interval  # Adjust based on interval
        if num_sequences <= 0:
            continue  # Skip users with insufficient data

        for i in range(num_sequences):
            start_idx = i * interval  # interval
            end_idx = start_idx + seq_len
            pred_start = end_idx
            pred_end = pred_start + pred_len

            # Ensure indices are within bounds
            if pred_end > len(user_data):
                continue

            # Input sequences
            seq_input_feat = input_features[start_idx:end_idx]  # [seq_len, 8]

            # Output sequences
            seq_output_feat = output_features[pred_start:pred_end]  # [pred_len, 4]
            seq_output_label = output_labels[pred_start:pred_end]  # [pred_len]

            # Append to lists
            input_seq_feature.append(seq_input_feat)
            predict_seq_feature.append(seq_output_feat)
            predict_seq_label.append(seq_output_label)  # [pred_len]

            # Append POI categories for input labels
            poi_categories_seq = user_data['Venue category name'].values[start_idx:end_idx]
            POI_categories.append(poi_categories_seq)

            # Append timestamps from start to end)
            timestamps.append(timestamps_user[start_idx: pred_end])



    # Convert lists to numpy arrays
    input_seq_feature = np.array(input_seq_feature, dtype=np.float32)      # [num_samples, seq_len, 8]
    predict_seq_feature = np.array(predict_seq_feature, dtype=np.float32)  # [num_samples, pred_len, 5]
    predict_seq_label = np.array(predict_seq_label, dtype=np.int64)        # [num_samples, pred_len]
    POI_categories = np.array(POI_categories, dtype=object)               # [num_samples, pred_len]
    timestamps = np.array(timestamps, dtype=object)                       # [num_samples, 2]
    

    print(f"Total sequences generated: {len(input_seq_feature)}")
    output_filename = f"foursquare_{city}_{'scaled' if scale else 'unscaled'}_size_{size[0]}_{size[1]}_{size[2]}.npz"
    # Save to .npz file
    np.savez_compressed(
        os.path.join(root_path, f"dataset/foursquare/{output_filename}"),
        input_seq_feature=input_seq_feature,
        predict_seq_feature=predict_seq_feature,
        predict_seq_label=predict_seq_label,
        POI_categories=POI_categories,
        timestamps=timestamps,
    )
    print(f"Preprocessed data saved to {output_filename}")
    
    
def preprocess_yj_data(root_path, city='D', scale=False, size=(7*48, 6*48, 1*48), interval=1*48, flag='train'):
    """
    Preprocesses the YjMob100k dataset and saves it as an .npz file.
    input_seq_feature: ['User ID', 'time slot', 'DayOfWeek', 'Day', 'Latitude', 'Longitude', 'Place ID']
    predict_seq_feature: ['User ID', 'time slot', 'DayOfWeek', 'Day']
    predict_seq_label: ['Place ID']

    Args:
        root_path (str): Root directory path where the dataset is located.
        city (str): City name, ['A', 'B', 'C', 'D']. 100k, 25k, 20k, 6k, -3k
        scale (bool): Whether to scale the numerical features.
        output_filename (str): Name of the output .npz file.
    """
    dataset = load_yj_df(city=city)
    total_users = dataset['uid'].nunique()
    
    samples_in_a_day = 48
    input_seq_length, _, predict_seq_length = size
    input_seq_length, predict_seq_length = input_seq_length // samples_in_a_day, predict_seq_length // samples_in_a_day
    print(f"Input sequence length: {input_seq_length} days, Prediction sequence length: {predict_seq_length} days")
    
    # flag = 'train' 'val' 'test', split the dataset 7:2:1 based on days (dataset['d'])
    total_days = dataset['d'].max() + 1
    if flag == 'train':
        start_day, end_day = 0, total_days * 0.7
    elif flag == 'val':
        start_day, end_day = total_days * 0.7 - input_seq_length, total_days * 0.9
    elif flag == 'test':
        start_day, end_day = total_days * 0.9 - input_seq_length, total_days
        
    if city == 'BOS': 
        if flag == 'train':
            start_day, end_day = 0, ceil(total_days * 0.6)
        elif flag == 'test':
            start_day, end_day = total_days - 14, total_days
        elif flag == 'val':
            start_day, end_day = ceil(total_days * 0.6) - 7, total_days - 7
    
    start_day = ceil(start_day)
    dataset = dataset[dataset['d'].between(start_day, end_day)]
    print(f"Total users: {total_users}, Total days: {total_days}, Total records: {len(dataset)}")
    print(f"Splitting data for {flag} set: {start_day} to {end_day}")

    day_values = np.arange(start_day, end_day)

    # Process sequences
    input_seq_feature = []
    predict_seq_feature = []
    predict_seq_label = []
    predict_seq_feature_full_7d = [] # For future use

    # Group data by user
    grouped_data = dataset.groupby('uid')
    print(f"Number of unique uid: {len(grouped_data)}")
    set_uids = np.arange(total_users)
    print(f"Missing uid is {set_uids[~np.in1d(set_uids, dataset['uid'].unique())]}") # 1785 4204

    for uid, uid_df in tqdm(grouped_data, desc="Generating sequences"):
        full_seq_x = generate_yj_sequence(uid_df, day_values)
        # Calculate the number of valid sequences
        future_days_to_save = 7
        max_future_days = max(predict_seq_length, future_days_to_save)
        num_seq = len(day_values) - input_seq_length - max_future_days + 1
        # num_seq = len(day_values) - input_seq_length - predict_seq_length + 1
        if num_seq > 0:
            # Precompute indices for slicing
            input_start_indices = np.arange(num_seq) * samples_in_a_day
            input_end_indices = input_start_indices + input_seq_length * samples_in_a_day
            predict_start_indices = input_end_indices
            predict_end_indices = predict_start_indices + predict_seq_length * samples_in_a_day
            predict_end_indices_7d = predict_start_indices + future_days_to_save * samples_in_a_day  

            # Slice sequences in bulk
            input_seq_feature.extend([full_seq_x[start:end] for start, end in zip(input_start_indices, input_end_indices)])
            predict_seq_feature.extend([full_seq_x[start:end, 0:4] for start, end in zip(predict_start_indices, predict_end_indices)])
            predict_seq_label.extend([full_seq_x[start:end, -1] for start, end in zip(predict_start_indices, predict_end_indices)])
            predict_seq_feature_full_7d.extend([full_seq_x[start:end, :] for start, end in zip(predict_start_indices, predict_end_indices_7d)])
    # Convert lists to numpy arrays and then to torch tensors
    print(f"Total sequences generated: {len(input_seq_feature)}")
    
    input_seq_feature = np.array(input_seq_feature, dtype=np.float32)
    predict_seq_feature = np.array(predict_seq_feature, dtype=np.float32)
    predict_seq_label = np.array(predict_seq_label, dtype=np.int64)
    predict_seq_feature_full_7d = np.array(predict_seq_feature_full_7d, dtype=np.float32)
    print(f"Future-7d full feature shape: {predict_seq_feature_full_7d.shape}")
    print(f"Input sequence shape: {input_seq_feature.shape}, Prediction sequence shape: {predict_seq_feature.shape}, Prediction label shape: {predict_seq_label.shape}")

    output_filename = f"yj_{city}_size_{size[0]}_{size[1]}_{size[2]}_{flag}.npz"
    np.savez_compressed(
        os.path.join(root_path, f"dataset/yj/{output_filename}"),
        input_seq_feature=input_seq_feature,
        predict_seq_feature=predict_seq_feature,
        predict_seq_label=predict_seq_label,
        predict_seq_feature_full_7d=predict_seq_feature_full_7d,
    )
    print(f"Preprocessed data saved to {output_filename}")
    return

def generate_yj_sequence(data_by_day, days):
    """
    Generate sequences for each user.

    Args:
        data_by_day (pd.DataFrame): Data for a specific user.
        days (list): List of unique days.

    Returns:
        np.ndarray: Full sequence features for the user.
    """
    uid = data_by_day['uid'].iloc[0]  # User ID
    time_steps = np.arange(48)  # 48 time slots in a day
    seq_x = []

    # Iterate over each day
    for d in days:
        # Initialize default values for the entire day
        full_day_x = np.full((48, 7), [-1, -1, -1, -1, 999, 999, 40000], dtype=np.float32)
        full_day_x[:, 0] = uid  # Set User ID
        full_day_x[:, 1] = time_steps  # Set time slots
        full_day_x[:, 2] = d % 7  # Day of the week (0-6)
        full_day_x[:, 3] = d  # Day

        # Check if the day exists in the user's data
        if d in data_by_day['d'].values:
            day_data = data_by_day[data_by_day['d'] == d].set_index('t')  # Set time as index

            # Extract valid time slots
            matching_times = day_data.index.values
            full_day_x[matching_times, -3] = day_data['x'].values  # Latitude
            full_day_x[matching_times, -2] = day_data['y'].values  # Longitude
            full_day_x[matching_times, -1] = day_data['label'].values  # Place ID

        seq_x.append(full_day_x)  # Add the day's data to the sequence

    return np.concatenate(seq_x, axis=0)  # Concatenate all days into one array
        
def yj_full4prompt(root_path='', city='D'):
    dataset = load_yj_df(city=city)
    
    uids = dataset['uid'].unique()
    
    full_seq = []
    
    for uid in tqdm(uids):
        uid_full_seq = generate_yj_sequence(dataset[dataset['uid'] == uid], np.arange(0, 75))
        uid_full_seq = uid_full_seq.reshape(75, 48, 7)
        full_seq.append(uid_full_seq)
    
    full_seq = np.array(full_seq, dtype=np.float32)
    print(f"Full sequence shape: {full_seq.shape}")
    
    # save as npy
    output_filename = f"yj_{city}_full_sequence.npy"
    np.save(os.path.join(root_path, f"dataset/yj/{output_filename}"), full_seq)
    return
    
def us_full4prompt(root_path='', city='BOS', num_users=10000):
    assert city in ['BOS', 'LA', "NYC"]
    data_path = os.path.join(root_path, f"dataset/us/{city}/test_df.csv")
    print(f"Loading data from {data_path}") 
    data = pd.read_csv(data_path, index_col=0).values
    data = data[:num_users]
    total_days = data.shape[1] // 48
    print(f"Total days: {total_days}")
    data = data.reshape(num_users, total_days, 48)
    print(f"Data shape: {data.shape}")
    
    with open(f'dataset/us/{city}/cbg_mapping.pkl', 'rb') as f:
        cbg_label_dict = pkl.load(f)
    cbg_demo_df = pd.read_csv('dataset/us/demo_geo_cbg_with_centroids.csv')
    if city == 'BOS':
        cbg_demo_df = cbg_demo_df[cbg_demo_df['NAMELSAD'] == "Boston-Cambridge-Newton, MA-NH Metro Area"]
    elif city == 'LA':
        cbg_demo_df = cbg_demo_df[cbg_demo_df['NAMELSAD'] == "Los Angeles-Long Beach-Anaheim, CA Metro Area"]
    elif city == 'NYC':
        cbg_demo_df = cbg_demo_df[cbg_demo_df['NAMELSAD'] == "New York-Newark-Jersey City, NY-NJ-PA Metro Area"]
    print(len(cbg_label_dict), cbg_demo_df.shape)
    reversed_dict = {v: k for k, v in cbg_label_dict.items()}
    
    # Pre-compute day_of_week array for all days
    days = np.arange(data.shape[1])
    days_of_week = (days + 3) % 7

    # Create meshgrid for all combinations
    uids, days, time_slots = np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing='ij'
    )

    # Create base array with uids, time_slots, and days_of_week
    base_array = np.stack([
        uids,
        time_slots,
        days_of_week[days],
        days,
        np.full_like(uids, 999, dtype=np.float32),  # default lat
        np.full_like(uids, 999, dtype=np.float32),  # default lon
        np.zeros_like(uids, dtype=np.float32)       # default location_id
    ], axis=-1)

    # Get mask for non-zero locations
    non_zero_mask = data > 0

    # Create location lookup array
    location_ids = data[non_zero_mask]
    geoids = np.array([reversed_dict[int(lid)] for lid in location_ids])
    coords = cbg_demo_df.set_index('CensusBlockGroup').loc[geoids, ['lat', 'lng']].values

    # Update the coordinates and location IDs where mask is True
    base_array[non_zero_mask, 4] = coords[:, 0]  # lat
    base_array[non_zero_mask, 5] = coords[:, 1]  # lon
    base_array[non_zero_mask, 6] = data[non_zero_mask]

    # Ensure final array has the correct dtype
    full_seq = base_array.astype(np.float32)
    print(f"Full sequence shape: {full_seq.shape}")

    # save as npy
    output_filename = f"us_{city}_full_sequence.npy"
    np.save(os.path.join(root_path, f"dataset/us/{city}/{output_filename}"), full_seq)
    return

def normalize_uids(df, uid_col='uid'):
    """
    Normalizes the 'uid' column in a DataFrame by making the IDs consecutive
    while maintaining the relative order.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a column containing non-consecutive integer UIDs.
    uid_col (str): The column name containing the UID values.

    Returns:
    pd.DataFrame: DataFrame with UIDs normalized to consecutive integers.
    """
    # Get unique, sorted UIDs
    unique_uids = sorted(df[uid_col].unique())

    # Create a mapping from old UID to new consecutive UID
    uid_mapping = {old_uid: new_uid for new_uid, old_uid in enumerate(unique_uids)}

    # Apply the mapping to the DataFrame
    df[uid_col] = df[uid_col].map(uid_mapping)

    return df

def load_yj_df(city='D'):
    '''
    City B remove uid [1785 4204] with incomplete data
    '''
    if city in ['B', 'C', 'D']:
        data_path = os.path.join(root_path, f"dataset/yj/tmp/city{city}_challengedata.csv.gz")
    elif city == 'A':
        data_path = os.path.join(root_path, "dataset/yj/cityA_groundtruthdata.csv.gz")
    elif city == 'BOS':
        data_path = os.path.join(root_path, f"dataset/yj/boston.csv.gz")
    else:
        raise ValueError("Invalid city name. Choose from ['A', 'B', 'C', 'D']")
    
    print(f"Loading data from {data_path}")
    dataset = pd.read_csv(data_path, compression='gzip')
    
    if city == 'B':
        dataset = dataset[~dataset['uid'].isin([1785, 4204, 13420, 304, 4214, 6984, 8502, 11557, 17781])]
    elif city == 'BOS':
        users_to_remove = [
            26, 54, 67, 69, 78, 85, 108, 137, 181, 196, 233, 234, 246, 249, 252, 255,
            315, 319, 367, 439, 444, 459, 461, 462, 483, 516, 520, 603, 612, 669,
            678, 687, 698, 747, 748, 752, 792, 807, 811, 818, 838, 845, 857, 861, 887, 921,
            961, 963, 998, 1007, 1020, 1049, 1072, 1094, 1104, 1149, 1203, 1257, 1307, 1322, 1346,
            1354, 1466, 1482, 1487, 1506, 1516, 1588, 1622, 1634, 1638, 1717, 1730, 1811, 1827,
            1840, 1846, 1870, 1871, 1887, 1895, 1934, 1956, 1984, 1987, 2008, 2020, 2022, 2048, 2057,
            2099, 2106, 2152, 2258, 2260, 2274, 2299, 2305, 2346, 2419, 2420, 2424, 2427, 2429, 2436,
            2469, 2493, 2501, 2510, 2514, 2516, 2563, 2572, 2594, 2649, 2655, 2674, 2678, 2681,
            2748, 2805, 2858, 2938, 2950, 2955, 2981, 3013, 3043, 3091, 3101, 3108, 3115, 3116,
            3221, 3242, 3245, 3287, 3313, 3358, 3381, 3387, 3448, 3479, 3499, 3500, 3506, 3513,
            3517, 3526, 3547, 3552, 3607, 3622, 3656, 3666, 3696, 3712, 3743, 3746, 3751, 3782,
            3792, 3808, 3814, 3835, 3847, 3854, 3864, 3883, 3893, 3956, 3969, 3982, 3983, 3986,
            3998, 4001, 4007, 4015, 4030, 4035, 4091, 4099, 4118, 4131, 4175, 4180, 4197, 4204,
            4220, 4238, 4243, 4267, 4289, 4373, 4395, 4410, 4414, 4425, 4470, 4507, 4508, 4549,
            4566, 4636, 4639, 4650, 4661, 4672, 4701, 4707, 4713, 4741, 4759, 4783, 4800, 4826, 4828,
            4835, 4857, 4860, 4872, 4882, 4886, 4888, 4948, 4964, 4979, 4986, 4987, 5007, 5026,
            5049, 5056, 5070, 5095, 5100, 5139, 5151, 5165, 5167, 5200, 5266, 5274, 5304, 5310,
            5345, 5381, 5388, 5403, 5418, 5451, 5467, 5478, 5499, 5500, 5541, 5549, 5561, 5588, 5590,
            5603, 5626, 5635, 5668, 5678, 5686, 5688, 5723, 5750, 5760, 5763, 5780, 5783, 5824, 5864, 5876, 5885,
            5888, 5889, 5900, 5914, 5938, 5939, 5958, 5999, 6034, 6064, 6132, 6134, 6138, 6161,
            6186, 6203, 6228, 6259, 6288, 6363, 6387, 6396, 6416, 6438, 6517, 6526, 6531, 6550,
            6566, 6587, 6590, 6599, 6694, 6697, 6706, 6715, 6759, 6768, 6786, 6794, 6806, 6821, 6848,
            6856, 6886, 6914, 6977, 7020, 7025, 7036, 7053, 7116, 7117, 7124, 7135, 7195, 7240, 7243, 7245,
            7259, 7286, 7295, 7351, 7377, 7391, 7395, 7410, 7417, 7435, 7440, 7444, 7470, 7497,
            7508, 7510, 7530, 7541, 7588, 7618, 7673, 7677, 7700, 7713, 7728, 7770, 7772, 7789, 7797,
            7812, 7849, 7883, 7891
        ]
        dataset = dataset[~dataset['uid'].isin(users_to_remove)]
        
    dataset = normalize_uids(dataset) # TODO: check if we need to do this or not
    # total_users = dataset['uid'].nunique() - (100000 - 100)
    
    if city in ['A', 'B', 'C', 'D']:
        total_users = dataset['uid'].nunique() - 3000 # Remove last 3000 users default 3000 # TODO: change here
        print(f"Total users: {total_users}")
        dataset = dataset[dataset['uid'] < total_users]
    else:
        total_users = dataset['uid'].nunique()
        print(f"Total users: {total_users}")
    
    dataset['label'] = 200 * (dataset['x'] - 1) + (dataset['y'] - 1)
    dataset.sort_values(by=['uid', 'd', 't'], inplace=True)
    
    return dataset
    
    

def preprocess_us_data(root_path='', city='BOS', size=(7*48, 6*48, 1*48), interval=1*48, flag='train'):
    data = np.load(f'dataset/us/{city}/us_{city}_full_sequence.npy')
    num_users = data.shape[0]
    total_days = data.shape[1]
    print(f"Total users: {num_users}, Total days: {total_days}")
    samples_in_a_day = 48
    
    if flag == 'train':
        start_day, end_day = 0, ceil(total_days * 0.6)
    elif flag == 'test':
        start_day, end_day = total_days - 14, total_days
    elif flag == 'val':
        start_day, end_day = ceil(total_days * 0.6) - 7, total_days - 7
        
    print(f"Splitting data for {flag} set: {start_day} to {end_day}")
    data = data[:, start_day:end_day]
    
    seq_len = size[0] // samples_in_a_day
    pred_len = size[2] // samples_in_a_day
    interval = interval // samples_in_a_day
    
    input_seq_feature = []
    predict_seq_feature = []
    predict_seq_label = []
    
    for uid in tqdm(range(num_users)):
        user_data = data[uid]
        
        num_sequences = (len(user_data) - seq_len - pred_len + 1) // interval
        for i in range(num_sequences):
            start_idx = i * interval
            end_idx = start_idx + seq_len 
            pred_start = end_idx
            pred_end = pred_start + pred_len
            
            if pred_end > len(user_data):
                continue
            
            seq_input_feat = user_data[start_idx:end_idx].reshape(-1, 7)
            seq_output_feat = user_data[pred_start:pred_end, :, :4].reshape(-1, 4)
            seq_output_label = user_data[pred_start:pred_end, :, -1].reshape(-1)
            
            input_seq_feature.append(seq_input_feat)
            predict_seq_feature.append(seq_output_feat)
            predict_seq_label.append(seq_output_label)
            
            
    input_seq_feature = np.array(input_seq_feature, dtype=np.float32)
    predict_seq_feature = np.array(predict_seq_feature, dtype=np.float32)
    predict_seq_label = np.array(predict_seq_label, dtype=np.int64)
    print(f"Input sequence shape: {input_seq_feature.shape}, Prediction sequence shape: {predict_seq_feature.shape}, Prediction label shape: {predict_seq_label.shape}")
    
    output_filename = f"us_{city}_size_{size[0]}_{size[1]}_{size[2]}_{flag}.npz"
    np.savez_compressed(
        os.path.join(root_path, f"dataset/us/{city}/{output_filename}"),
        input_seq_feature=input_seq_feature,
        predict_seq_feature=predict_seq_feature,
        predict_seq_label=predict_seq_label,
    )

    return

if __name__ == "__main__":
    # seq_len, label_len, pred_len = 40, 35, 5 # 20, 15, 5
    # size = (seq_len, label_len, pred_len)
    # root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # preprocess_foursquare_data(root_path, city='NYC', scale=False, size=size, interval=None)
    # preprocess_foursquare_data(root_path, city='TKY', scale=False, size=size, interval=None)
    
    seq_len, label_len, pred_len = 7*48, 6*48, 1*48
    interval = 1*48
    size = (seq_len, label_len, pred_len)
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
    city = 'C'
    yj_full4prompt(root_path, city=city)
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='train')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='val')
    preprocess_yj_data(root_path, city=city, scale=False, size=size, interval=interval, flag='test')
    
    # city = 'LA'
    # # us_full4prompt(root_path, city=city, num_users=10000)
    # preprocess_us_data(root_path, city=city, size=size, interval=interval, flag='train')
    # preprocess_us_data(root_path, city=city, size=size, interval=interval, flag='val')
    # preprocess_us_data(root_path, city=city, size=size, interval=interval, flag='test')
