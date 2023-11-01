import pandas as pd
import numpy as np
from faker import Faker

class DataGenerator:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.fake = Faker()
    
    def generate_data(self):
        src_ips = [self.fake.ipv4() for _ in range(self.n_samples)]
        dest_ips = [self.fake.ipv4() for _ in range(self.n_samples)]
        
        timestamps = pd.date_range(start='2022-01-01', periods=self.n_samples, freq='T')
        
        # Temporal Patterns: More TCP during "business hours" (9 AM - 5 PM)
        protocols = np.where(
            (timestamps.hour >= 9) & (timestamps.hour <= 17),
            'TCP',
            np.random.choice(['UDP', 'ICMP'], self.n_samples)
        )
        
        # Port-Protocol coherence: Assign ports based on protocol
        src_ports = np.random.randint(1024, 65535, self.n_samples)
        dest_ports = np.where(
            protocols == 'TCP',
            np.random.choice([80, 443], self.n_samples),
            np.random.choice([22, 21], self.n_samples)
        )
        
        is_anomalous = np.random.choice([0, 1], self.n_samples, p=[0.9, 0.1])
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'src_ip': src_ips,
            'dest_ip': dest_ips,
            'protocol': protocols,
            'src_port': src_ports,
            'dest_port': dest_ports,
            'is_anomalous': is_anomalous
        })
        
        return df