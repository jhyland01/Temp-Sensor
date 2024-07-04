import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.original_cols = [
    '1_Wind direction (°)',
       '1_Nacelle position (°)', '1_Power (kW)',
       '1_Generator RPM (RPM)',
       ]
        self.eracols = ['2_m_temp_mean_deg-c',
       '2_m_dew_point_temp_deg-c', 'friction_velocity_mps']

    def create_features(self):
        self.data['avg_pitch_angle'] = (self.data['1_Blade angle (pitch position) A (°)'] + self.data['1_Blade angle (pitch position) B (°)'] + self.data['1_Blade angle (pitch position) C (°)']) / 3

    def add_change(self):
        for col in self.original_cols:
            self.data[f'{col}_change'] = self.data[col].diff()
    
    @staticmethod
    def entropy(series):
        value_counts = series.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts))
        return entropy
    
    @staticmethod
    def seasonality_features(df):
        # df['month_sin'] = np.sin(2*np.pi*df.index.month/12)
        df['month_cos'] = np.cos(2*np.pi*df.index.month/12)*-1
        # df['hour_sin'] = np.sin(2*np.pi*df.index.hour/24)
        df['hour_cos'] = np.cos(2*np.pi*df.index.hour/24)*-1
        df['cos_wd'] = np.cos(2*np.pi*df['1_Wind direction (°)']/360)
        df['cos_np'] = np.cos(2*np.pi*df['1_Nacelle position (°)']/360)
        df['avg_dir'] = (df['cos_wd'] + df['cos_np'])/2
        return df
    
    def adjust_ws(self):
        # set to zero if less than 3 or greater than 24
        self.data.loc[self.data['1_Wind speed (m/s)'] < 3, '1_Wind speed (m/s)'] = 0
        self.data.loc[self.data['1_Wind speed (m/s)'] > 24, '1_Wind speed (m/s)'] = 0
    
    def calc_seasons(self):
        self.data = self.seasonality_features(self.data)
    
    @staticmethod
    def crest_factor(series):
        return series.max() / np.sqrt(np.mean(np.square(series)))
    
    def calc_stats(self):
        for col in self.original_cols:
            window = 50
            self.data[f'{col}_rollmean'] = self.data[col].rolling(window=window, center=False).mean()
            self.data[f'{col}_rollstd'] = self.data[col].rolling(window=window, center=False).std()
            self.data[f'{col}_crest_factor'] = self.data[col].rolling(window=window, center=False).apply(self.crest_factor)
            # self.data[f'{col}_kurtosis'] = self.data[col].rolling(window=6, center=True).kurt()

    def add_curtailed(self):
        self.data['curtailed'] = (self.data['1_Wind speed (m/s)'] >= 11) & ((self.data['1_Generator RPM (RPM)'] >= 850) & (self.data['1_Generator RPM (RPM)'] <= 1100))

    def add_cooling(self):
        self.data['cooling'] = (self.data['1_Generator RPM (RPM)'] >= 1600)

    def add_ti(self):
        self.data['rollti'] = (
            self.data['1_Wind speed (m/s)'].rolling(window=12, center=True).std()
            / self.data['1_Wind speed (m/s)']
            .rolling(window=12, center=True)
            .mean()
        )
        
    def add_offline(self):
        self.data['offline'] = self.data['1_Generator RPM (RPM)'] <= 1000

    def add_historic_offline(self):
        self.data['historic_offline'] = self.data['offline'].rolling(window=6).mean()

    def add_best_lag_or_lead_features(self, best_lag_or_lead_dict):
        for df in [self.data]:
            for feature, (corr_type, steps) in best_lag_or_lead_dict.items():
                new_column_name = f"{feature}_{corr_type}{steps}"
                
                if corr_type == 'lag':
                    df[new_column_name] = df[feature].shift(steps)
                elif corr_type == 'lead':
                    df[new_column_name] = df[feature].shift(-steps)

    def convert_and_set_index(self):
        self.data['# Date and time'] = pd.to_datetime(self.data['# Date and time'])
        self.data.set_index('# Date and time', inplace=True)

    def impute(self):
        imputer = KNNImputer(n_neighbors=5)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns, index=self.data.index)

    def drop_columns(self):
        columns = [
            # '1_Wind speed (m/s)',
            '1_Rotor speed (RPM)',
            '1_Current L1 / U (A)', '1_Current L2 / V (A)', '1_Current L3 / W (A)', '1_Apparent power (kVA)',
            '1_Gearbox speed (RPM)',
            '1_Yaw bearing angle (°)',
            '1_Reactive power (kvar)',
        ]
        self.data.drop(columns, axis=1, inplace=True)
        self.data.drop(columns=[col for col in self.data.columns if 'Voltage' in col or 'position)' in col], inplace=True)


    def process_all(self):
        self.add_change()
        # self.add_curtailed()
        self.add_offline()
        self.add_historic_offline()
        self.adjust_ws()
        self.add_cooling()
        self.add_ti()
        self.create_features()
        self.calc_stats()
        self.add_best_lag_or_lead_features({
            '1_Wind direction (°)': ['lead', 6], 
            '1_Nacelle position (°)': ['lead', 3], 
            '1_Power (kW)': ['lag', 6], 
            '1_Generator RPM (RPM)': ['lead', 6],
        })
        self.drop_columns()
        self.convert_and_set_index()
        self.calc_seasons()
        self.impute()

# # Usage
# train = pd.read_csv('data/train.csv')  # Replace with your actual paths
# test = pd.read_csv('data/test.csv')    # Replace with your actual paths

# processor = DataProcessor(train, test)
# processor.process_all()

# # Now `processor.train` and `processor.test` will contain all the additional features.
