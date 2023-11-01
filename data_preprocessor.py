from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def handle_missing_values(self):
        # For this example, we'll just drop missing values
        self.df.dropna(inplace=True)

    def encode_categorical_features(self):
        le = LabelEncoder()
        self.df['protocol'] = le.fit_transform(self.df['protocol'])
        self.df['src_ip'] = le.fit_transform(self.df['src_ip'])
        self.df['dest_ip'] = le.fit_transform(self.df['dest_ip'])

    def normalize_features(self):
        # May not be necessary for tree-based models
        scaler = StandardScaler()
        self.df[['src_port', 'dest_port']] = scaler.fit_transform(self.df[['src_port', 'dest_port']])

    def preprocess(self):
        self.handle_missing_values()
        self.encode_categorical_features()
        self.normalize_features()
        return self.df