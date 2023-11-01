from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from feature_engineering import engineer_datetime_features
from joblib import dump, load
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV



class GBMTrainer:
    def __init__(self, df):
        self.df = engineer_datetime_features(df)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def split_data(self):
        X = self.df.drop('is_anomalous', axis=1)
        y = self.df['is_anomalous']
        sm = SMOTE(random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def train_model(self):
        self.model = GradientBoostingClassifier()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        if sum(y_pred) == 0:
            print("Model has not predicted any positive samples.")
            return

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=1)
        recall = recall_score(self.y_test, y_pred, zero_division=1)
        f1 = f1_score(self.y_test, y_pred, zero_division=1)

        print(f"Model Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    def tune_model(self):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3]
        }
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)
        # Interesting note - in scikit-learning, the underscore trailing the attribute indicates it's been fitted with data
        self.model = grid_search.best_estimator_
        print("Best hyperparameters found: ", grid_search.best_params_)
        print("Assigning the best model to self.model")

    def save_model(self, filename):
        print(f"Saving model to {filename}")
        dump(self.model, filename)
        
    def load_model(self, filename):
        print(f"Loading model from {filename}")
        self.model = load(filename)
        
    def execute(self):
        print("Splitting data...")
        self.split_data()
        print("Data split done.")
        
        print("Training model...")
        self.train_model()
        print("Model trained.")

        print("Tuning model...")
        self.tune_model()
        print("Model tuning done.")
        
        print("Evaluating model...")
        self.evaluate_model()
        print("Model evaluation done.")
        
        print("Saving model...")
        self.save_model('gbm_model.joblib')
        print("Model saved.")

        print("Creating visualization of the importance of each feature...")
        self.plot_feature_importance()
        print("Model saved.")

        print("Calculating model performance:")
        self.calculate_metrics()

        print("Plotting the confusion matrix...")
        self.plot_confusion_matrix()

    def plot_feature_importance(self):
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.figure(figsize=(12, 6))
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(self.df.columns)[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.show()

    def calculate_metrics(self):
        y_pred = self.model.predict(self.X_test)
        print(f"Precision: {precision_score(self.y_test, y_pred)}")
        print(f"Recall: {recall_score(self.y_test, y_pred)}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred)}")
        
    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()