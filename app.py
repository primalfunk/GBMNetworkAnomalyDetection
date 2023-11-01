from data_generator import DataGenerator
from data_preprocessor import DataPreprocessor
from gbm_trainer import GBMTrainer

# Generate Data
data_gen = DataGenerator(n_samples=1000)
df = data_gen.generate_data()

# Summarize and Display the head of the Generated Data
print("Summary of Generated Data:")
print(df.describe())
print("Head of Generated Data:")
print(df.head())

# Preprocess Data
preprocessor = DataPreprocessor(df)
processed_df = preprocessor.preprocess()

# Summarize and Display the head of the Preprocessed Data
print("Summary of Preprocessed Data:")
print(processed_df.describe())
print("Head of Preprocessed Data:")
print(processed_df.head())

# Train and Evaluate GBM Model
gbm_trainer = GBMTrainer(processed_df)
gbm_trainer.execute()



