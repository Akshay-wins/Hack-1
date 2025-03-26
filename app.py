import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import scipy.signal as signal

class AdvancedSideChannelAnalyzer:
    def __init__(self, num_samples=1000, trace_length=100):
        """
        Advanced Side-Channel Vulnerability Analyzer
        
        Args:
        - num_samples: Number of synthetic traces to generate
        - trace_length: Length of each trace
        """
        self.num_samples = num_samples
        self.trace_length = trace_length
        
        # Data storage
        self.traces = None
        self.labels = None
        self.preprocessed_traces = None
        
        # Models
        self.neural_network_model = None
        self.random_forest_model = None
        
        # Analysis results
        self.analysis_report = {}

    def generate_synthetic_data(self):
        """
        Generate advanced synthetic side-channel traces with multiple vulnerability indicators
        
        Returns:
        - traces: Synthetic power traces
        - labels: Corresponding vulnerability labels
        """
        np.random.seed(42)
        
        # Generate base traces with multiple complexity levels
        traces = np.random.randn(self.num_samples, self.trace_length)
        
        # Create multi-class labels based on vulnerability levels
        labels = np.zeros(self.num_samples, dtype=int)
        labels[np.sum(traces, axis=1) > np.percentile(np.sum(traces, axis=1), 75)] = 1
        labels[np.sum(traces, axis=1) > np.percentile(np.sum(traces, axis=1), 90)] = 2
        
        # Simulate different vulnerability scenarios
        for i in range(self.num_samples):
            # Ensure more distinct characteristics for class 1
            if labels[i] == 1:
                # Add more distinctive features for moderate risk
                traces[i] += np.sin(np.linspace(0, 10, self.trace_length)) * 1.5
                traces[i] += np.random.normal(0, 0.5, self.trace_length)
            
            # Sinusoidal variations to mimic power consumption
            traces[i] += 0.5 * np.sin(np.linspace(0, 10, self.trace_length))
            
            # Simulate different attack-relevant signal characteristics
            # 1. Timing variation
            if i % 5 == 0:
                traces[i] *= np.linspace(0.8, 1.2, self.trace_length)
            
            # 2. Electromagnetic emission simulation
            if i % 7 == 0:
                traces[i] += signal.chirp(np.linspace(0, 1, self.trace_length), 0.1, 1, 10)
            
            # 3. Power analysis spike indicators
            spike_indices = np.random.choice(self.trace_length, 3, replace=False)
            traces[i, spike_indices] += np.random.uniform(2, 5, 3)
        
        self.traces = traces
        self.labels = labels
        
        return traces, labels

    def preprocess_data(self, test_size=0.2):
        """
        Preprocess traces with advanced feature extraction
        
        Returns:
        - Preprocessed train and test splits
        """
        # Standardization
        scaler = StandardScaler()
        traces_scaled = scaler.fit_transform(self.traces)
        
        # Feature engineering
        def extract_features(traces):
            features = []
            for trace in traces:
                # Statistical features
                trace_features = [
                    np.mean(trace),
                    np.std(trace),
                    np.max(trace),
                    np.min(trace),
                    np.percentile(trace, 25),
                    np.percentile(trace, 75),
                ]
                
                # Frequency domain features
                fft_trace = np.abs(np.fft.fft(trace))
                trace_features.extend([
                    np.mean(fft_trace[:len(fft_trace)//2]),
                    np.max(fft_trace[:len(fft_trace)//2])
                ])
                
                features.append(trace_features)
            
            return np.array(features)
        
        # Extract advanced features
        traces_features = extract_features(traces_scaled)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            traces_features, self.labels, 
            test_size=test_size, 
            random_state=42,
            stratify=self.labels  # Ensure balanced class distribution
        )
        
        self.preprocessed_traces = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        return X_train, X_test, y_train, y_test

    def create_neural_network_model(self, input_shape):
        """
        Create advanced neural network for vulnerability detection
        """
        # Compute class weights to focus on precision
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(self.preprocessed_traces['y_train']), 
            y=self.preprocessed_traces['y_train']
        )
        class_weights = dict(enumerate(class_weights))
        
        # Boost weight for precision of class 1
        class_weights[1] *= 1.5
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model, class_weights

    def train_models(self):
        """
        Train multiple models for comprehensive vulnerability analysis
        """
        # Ensure data is preprocessed
        if self.preprocessed_traces is None:
            self.generate_synthetic_data()
            self.preprocess_data()
        
        X_train, X_test, y_train, y_test = (
            self.preprocessed_traces['X_train'],
            self.preprocessed_traces['X_test'],
            self.preprocessed_traces['y_train'],
            self.preprocessed_traces['y_test']
        )
        
        # Neural Network Model
        self.neural_network_model, class_weights = self.create_neural_network_model((X_train.shape[1],))
        nn_history = self.neural_network_model.fit(
            X_train, y_train, 
            validation_split=0.2,
            epochs=50, 
            batch_size=32, 
            verbose=0,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                )
            ]
        )
        
        # Random Forest Model
        # Compute class weights for Random Forest
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        
        self.random_forest_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight=class_weights,
            # Focus on improving class 1 performance
            min_samples_leaf=3  # Reduce overfitting for minority class
        )
        self.random_forest_model.fit(X_train, y_train)
        
        # Evaluate Models
        nn_predictions = np.argmax(self.neural_network_model.predict(X_test), axis=1)
        rf_predictions = self.random_forest_model.predict(X_test)
        
        # Comprehensive Analysis Report
        self.analysis_report = {
            'Neural Network': {
                'Accuracy': self.neural_network_model.evaluate(X_test, y_test)[1],
                'Classification Report': classification_report(y_test, nn_predictions, output_dict=True)
            },
            'Random Forest': {
                'Accuracy': self.random_forest_model.score(X_test, y_test),
                'Classification Report': classification_report(y_test, rf_predictions, output_dict=True)
            }
        }
        
        return self.analysis_report

    def generate_vulnerability_insights(self):
        """
        Generate detailed vulnerability insights and potential countermeasures
        """
        X_test = self.preprocessed_traces['X_test']
        y_test = self.preprocessed_traces['y_test']
        
        # Get predictions
        nn_predictions = np.argmax(self.neural_network_model.predict(X_test), axis=1)
        rf_predictions = self.random_forest_model.predict(X_test)
        
        # Vulnerability Hotspot Analysis
        vulnerability_insights = {
            'Vulnerability Levels': {
                0: 'Low Risk',
                1: 'Moderate Risk',
                2: 'High Risk'
            },
            'Risk Distribution': dict(zip(*np.unique(y_test, return_counts=True))),
            'Misclassification Patterns': {
                'Neural Network': confusion_matrix(y_test, nn_predictions).tolist(),
                'Random Forest': confusion_matrix(y_test, rf_predictions).tolist()
            },
            'Countermeasure Recommendations': {
                0: [
                    "Standard cryptographic protocols",
                    "Basic side-channel resistance techniques"
                ],
                1: [
                    "Enhanced masking techniques",
                    "Dynamic frequency scaling",
                    "Noise injection in power traces"
                ],
                2: [
                    "Complete hardware redesign",
                    "Advanced differential power analysis countermeasures",
                    "Comprehensive electromagnetic shielding"
                ]
            }
        }
        
        return vulnerability_insights

    def visualize_results(self):
        """
        Comprehensive visualization of analysis results
        
        Returns:
        - fig: Matplotlib figure with visualizations
        """
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Trace Characteristics
        plt.subplot(2, 3, 1)
        plt.title('Sample Traces by Vulnerability Level')
        for level in [0, 1, 2]:
            sample_traces = self.traces[self.labels == level][:5]
            for trace in sample_traces:
                plt.plot(trace, alpha=0.5, label=f'Level {level}')
        plt.legend()
        
        # 2. Confusion Matrix for Neural Network
        plt.subplot(2, 3, 2)
        nn_predictions = np.argmax(self.neural_network_model.predict(self.preprocessed_traces['X_test']), axis=1)
        cm = confusion_matrix(self.preprocessed_traces['y_test'], nn_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Neural Network Confusion Matrix')
        
        # 3. ROC Curve
        plt.subplot(2, 3, 3)
        nn_probs = self.neural_network_model.predict(self.preprocessed_traces['X_test'])
        roc_auc_scores = {}
        for i in range(3):
            fpr, tpr, _ = roc_curve((self.preprocessed_traces['y_test'] == i).astype(int), nn_probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[i] = roc_auc
            plt.plot(fpr, tpr, label=f'Level {i} (AUC = {roc_auc:.2f})')
        plt.title('Multiclass ROC Curve')
        plt.legend()
        
        plt.tight_layout()
        return fig


class SideChannelAttackSimulator:
    def __init__(self, analyzer):
        """
        Simulate advanced side-channel attacks
        
        Args:
        - analyzer: AdvancedSideChannelAnalyzer instance
        """
        self.analyzer = analyzer
        self.attack_traces = None
        self.attack_labels = None
    
    def generate_attack_traces(self, attack_type='power'):
        """
        Generate synthetic attack traces simulating different side-channel vulnerabilities
        
        Args:
        - attack_type: Type of side-channel attack ('power', 'timing', 'electromagnetic')
        
        Returns:
        - Simulated attack traces and corresponding labels
        """
        np.random.seed(42)
        
        # Use existing trace generation method as base
        traces, labels = self.analyzer.generate_synthetic_data()
        
        # Modify traces to simulate specific attack characteristics
        if attack_type == 'power':
            # Simulate power analysis attack with more pronounced variations
            for i in range(len(traces)):
                traces[i] *= np.sin(np.linspace(0, np.pi, len(traces[i]))) + 1
                traces[i] += np.random.normal(0, 0.5, len(traces[i]))
                
                # Introduce artificial power leakage indicators
                if labels[i] > 0:
                    traces[i] += np.random.uniform(1, 3, len(traces[i]))
        
        elif attack_type == 'timing':
            # Simulate timing-based attack with execution time variations
            for i in range(len(traces)):
                timing_variation = np.linspace(0.8, 1.2, len(traces[i]))
                traces[i] *= timing_variation
                
                # More pronounced variations for higher vulnerability levels
                traces[i] += labels[i] * np.random.uniform(0.5, 2, len(traces[i]))
        
        elif attack_type == 'electromagnetic':
            # Simulate electromagnetic emission-based attack
            for i in range(len(traces)):
                # Generate chirp signals with varying intensity
                chirp_signal = signal.chirp(
                    np.linspace(0, 1, len(traces[i])), 
                    0.1, 1, labels[i] * 10
                )
                traces[i] += chirp_signal
                traces[i] += np.random.normal(0, 0.3, len(traces[i]))
        
        self.attack_traces = traces
        self.attack_labels = labels
        
        return traces, labels
    
    def extract_features(self, traces):
        """
        Extract advanced features from traces
        
        Args:
        - traces: Input traces
        
        Returns:
        - Extracted features as numpy array
        """
        features = []
        for trace in traces:
            # Statistical features
            trace_features = [
                np.mean(trace),
                np.std(trace),
                np.max(trace),
                np.min(trace),
                np.percentile(trace, 25),
                np.percentile(trace, 75),
            ]
            
            # Frequency domain features
            fft_trace = np.abs(np.fft.fft(trace))
            trace_features.extend([
                np.mean(fft_trace[:len(fft_trace)//2]),
                np.max(fft_trace[:len(fft_trace)//2])
            ])
            
            features.append(trace_features)
        
        return np.array(features)
    
    def evaluate_attack_detection(self):
        """
        Evaluate attack detection performance
        
        Returns:
        - Comprehensive attack detection report
        """
        # Vulnerability levels definition
        vulnerability_levels = {
            0: "Low Vulnerability",
            1: "Moderate Vulnerability",
            2: "High Critical Vulnerability"
        }
        
        # Preprocess attack traces
        scaler = StandardScaler()
        attack_traces_scaled = scaler.fit_transform(self.attack_traces)
        
        # Extract features for attack traces
        attack_features = self.extract_features(attack_traces_scaled)
        
        # Get predictions from both models
        nn_predictions = np.argmax(self.analyzer.neural_network_model.predict(attack_features), axis=1)
        rf_predictions = self.analyzer.random_forest_model.predict(attack_features)
        
        def calculate_vulnerability_score(predictions, true_labels):
            """
            Calculate detailed vulnerability metrics
            
            Args:
            - predictions: Model predictions
            - true_labels: Actual vulnerability labels
            
            Returns:
            - Comprehensive vulnerability score dictionary
            """
            high_risk_traces = np.sum(true_labels == 2)
            correctly_detected_high_risk = np.sum((predictions == 2) & (true_labels == 2))
            
            vulnerability_score = {
                'Total High-Risk Traces': high_risk_traces,
                'Correctly Detected High-Risk Traces': correctly_detected_high_risk,
                'Detection Rate': (correctly_detected_high_risk / high_risk_traces) * 100 if high_risk_traces > 0 else 0,
                'Vulnerability Level': vulnerability_levels[2] if correctly_detected_high_risk / high_risk_traces < 0.5 else "Critical Vulnerability",
                'Confusion Matrix': confusion_matrix(true_labels, predictions).tolist(),
                'Classification Report': classification_report(true_labels, predictions, output_dict=True)
            }
            
            return vulnerability_score
        
        # Calculate vulnerability for both models
        attack_report = {
            'Neural Network': calculate_vulnerability_score(nn_predictions, self.attack_labels),
            'Random Forest': calculate_vulnerability_score(rf_predictions, self.attack_labels)
        }
        
        return attack_report
    
    def visualize_attack_detection(self, attack_report):
        """
        Create visualization for attack detection results
        
        Args:
        - attack_report: Comprehensive attack detection results
        
        Returns:
        - Matplotlib figure with attack visualization
        """
        fig = plt.figure(figsize=(20, 10))
        
        # Neural Network Confusion Matrix
        plt.subplot(1, 2, 1)
        nn_cm = attack_report['Neural Network']['Confusion Matrix']
        sns.heatmap(nn_cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Neural Network: Attack Detection Confusion Matrix')
        plt.xlabel('Predicted Vulnerability Level')
        plt.ylabel('Actual Vulnerability Level')
        
        # Random Forest Confusion Matrix
        plt.subplot(1, 2, 2)
        rf_cm = attack_report['Random Forest']['Confusion Matrix']
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Random Forest: Attack Detection Confusion Matrix')
        plt.xlabel('Predicted Vulnerability Level')
        plt.ylabel('Actual Vulnerability Level')
        
        plt.tight_layout()
        return fig


def main():
    # Streamlit App
    st.title("🔒 Side-Channel Vulnerability Analyzer")
    
    # Sidebar for configuration
    st.sidebar.header("Analysis Configuration")
    num_samples = st.sidebar.slider("Number of Samples", 500, 5000, 2000)
    trace_length = st.sidebar.slider("Trace Length", 50, 500, 200)
    
    # Initialize Analyzer
    analyzer = AdvancedSideChannelAnalyzer(
        num_samples=num_samples,
        trace_length=trace_length
    )
    
    # Attack Simulation Configuration
    st.sidebar.header("Attack Simulation")
    attack_types = st.sidebar.multiselect(
        "Select Attack Types", 
        ['Power', 'Timing', 'Electromagnetic'], 
        default=['Power']
    )
    
    # Main Analysis Flow
    if st.button("Run Side-Channel Analysis"):
        # Generate Synthetic Data
        with st.spinner("Generating Synthetic Side-Channel Traces..."):
            traces, labels = analyzer.generate_synthetic_data()
            st.success(f"Generated {num_samples} synthetic traces")
        
        # Preprocess Data
        with st.spinner("Preprocessing Traces..."):
            analyzer.preprocess_data()
            st.success("Data Preprocessed")
        
        # Train Models
        with st.spinner("Training Vulnerability Detection Models..."):
            analysis_report = analyzer.train_models()
            st.success("Models Trained Successfully")
        
        # Display Model Performance
        st.subheader("📊 Model Performance")
        for model_name, report in analysis_report.items():
            st.write(f"{model_name} Model:")
            st.write(f"  Accuracy: {report['Accuracy']*100:.2f}%")
        
        # Generate Vulnerability Insights
        st.subheader("🔒 Vulnerability Insights")
        vulnerability_report = analyzer.generate_vulnerability_insights()
        
        # Risk Distribution
        st.write("Risk Distribution:")
        for level, count in vulnerability_report['Risk Distribution'].items():
            st.write(f"  Level {level}: {count} traces")
        
        # Countermeasure Recommendations
        st.subheader("🛡 Countermeasure Recommendations")
        for level, recommendations in vulnerability_report['Countermeasure Recommendations'].items():
            st.write(f"Risk Level {level}:")
            for rec in recommendations:
                st.write(f"- {rec}")
        
        # Visualizations
        st.subheader("📈 Analysis Visualizations")
        fig = analyzer.visualize_results()
        st.pyplot(fig)
        
        # Simulate Side-Channel Attacks
        st.subheader("🚨 Side-Channel Attack Simulation")
        attack_simulator = SideChannelAttackSimulator(analyzer)
        
        for attack_type in [t.lower() for t in attack_types]:
            st.write(f"\n--- {attack_type.capitalize()} Side-Channel Attack Simulation ---")
            
            # Generate attack traces
            attack_traces, attack_labels = attack_simulator.generate_attack_traces(attack_type)
            
            # Evaluate attack detection
            attack_report = attack_simulator.evaluate_attack_detection()
            
            # Display attack detection results
            st.write("\nAttack Detection Performance:")
            for model_name, report in attack_report.items():
                st.write(f"{model_name} Performance:")
                st.write(report['Classification Report'])
            
            # Visualize attack detection
            attack_visualization = attack_simulator.visualize_attack_detection(attack_report)
            st.pyplot(attack_visualization)


if __name__ == '__main__':
    main()