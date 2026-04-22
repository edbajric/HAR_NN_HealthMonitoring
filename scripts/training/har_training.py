"""
Human Activity Recognition (HAR) - MLP Training Script

This script implements the complete Experimentation Cycle for training a
Multilayer Perceptron (MLP) on the UCI HAR dataset. It follows the scientific
methodology for building a Rational Agent that classifies physical activity.

Architecture: 561 -> 256 -> 128 -> 64 -> 6 (Softmax)

Key Features:
- Proper data scaling using ONLY training set parameters (no data leakage)
- He initialization for ReLU layers
- Deterministic seed for reproducibility
- Early stopping on validation loss
- Cross-entropy loss with epsilon for numerical stability
- Adam optimizer with configurable learning rate

Usage:
    python har_training.py --train_path ../data/train.csv --test_path ../data/test.csv

Author: HAR Agent Project
"""

import numpy as np
import json
import argparse
from typing import Tuple, Dict, List
from pathlib import Path

# Set deterministic seed for scientific reproducibility
SEED = 42
np.random.seed(SEED)

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 10
EPSILON = 1e-7

# Activity labels
ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
NUM_CLASSES = 6
INPUT_SIZE = 561


class DenseLayer:
    """
    Fully connected (dense) layer implementing forward and backward passes.
    
    Each neuron computes: output = activation(weights @ input + bias)
    """
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # He initialization for ReLU, Glorot for softmax
        if activation == 'relu':
            # He initialization: variance = 2/fan_in
            stddev = np.sqrt(2.0 / input_size)
        else:
            # Glorot initialization: variance = 2/(fan_in + fan_out)
            stddev = np.sqrt(2.0 / (input_size + output_size))
        
        self.weights = np.random.randn(input_size, output_size) * stddev
        self.biases = np.zeros((1, output_size))
        
        # Gradient storage for backpropagation
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        
        # Cache for backward pass
        self.inputs = None
        self.output = None
        self.activation_output = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass: linear transformation + activation"""
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases
        
        # Apply activation
        if self.activation == 'relu':
            self.activation_output = self.relu(self.output)
        elif self.activation == 'softmax':
            self.activation_output = self.softmax(self.output)
        else:
            self.activation_output = self.output
        
        return self.activation_output
    
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradients using chain rule"""
        # Activation gradient
        if self.activation == 'relu':
            dvalues = dvalues * self.relu_derivative(self.output)
        # Note: softmax gradient is combined with cross-entropy in loss function
        
        # Gradients for weights and biases
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient for previous layer (input gradient)
        self.dinputs = dvalues @ self.weights.T
        
        return self.dinputs
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative: 1 if x > 0, else 0"""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax activation: converts logits to probability distribution.
        Numerically stable version with max subtraction.
        """
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / (np.sum(exp_values, axis=1, keepdims=True) + EPSILON)


class CrossEntropyLoss:
    """
    Categorical Cross-Entropy Loss with numerical stability.
    
    Combined with softmax gradient for efficient backpropagation.
    """
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate cross-entropy loss.
        
        Args:
            predictions: Softmax probabilities (batch_size, num_classes)
            targets: One-hot encoded labels (batch_size, num_classes)
        
        Returns:
            Mean loss over batch
        """
        # Clip predictions to avoid log(0)
        clipped = np.clip(predictions, EPSILON, 1 - EPSILON)
        
        # Cross-entropy: -sum(y * log(y_hat))
        losses = -np.sum(targets * np.log(clipped), axis=1)
        
        return np.mean(losses)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Combined softmax + cross-entropy gradient.
        
        This simplifies to: dL/dz = predictions - targets
        """
        samples = predictions.shape[0]
        return (predictions - targets) / samples


class AdamOptimizer:
    """
    Adam optimizer with momentum and adaptive learning rates.
    
    Combines benefits of AdaGrad and RMSProp for faster convergence.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMSProp)
    
    def update(self, layer: DenseLayer, layer_id: str):
        """Update layer weights using Adam algorithm"""
        self.t += 1
        
        # Initialize moments if needed
        if layer_id not in self.m:
            self.m[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            self.v[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
        
        # Update moments for weights
        self.m[layer_id]['weights'] = (self.beta1 * self.m[layer_id]['weights'] + 
                                        (1 - self.beta1) * layer.dweights)
        self.v[layer_id]['weights'] = (self.beta2 * self.v[layer_id]['weights'] + 
                                        (1 - self.beta2) * layer.dweights**2)
        
        # Bias correction
        m_hat_w = self.m[layer_id]['weights'] / (1 - self.beta1**self.t)
        v_hat_w = self.v[layer_id]['weights'] / (1 - self.beta2**self.t)
        
        # Update weights
        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        
        # Update moments for biases
        self.m[layer_id]['biases'] = (self.beta1 * self.m[layer_id]['biases'] + 
                                       (1 - self.beta1) * layer.dbiases)
        self.v[layer_id]['biases'] = (self.beta2 * self.v[layer_id]['biases'] + 
                                       (1 - self.beta2) * layer.dbiases**2)
        
        # Bias correction
        m_hat_b = self.m[layer_id]['biases'] / (1 - self.beta1**self.t)
        v_hat_b = self.v[layer_id]['biases'] / (1 - self.beta2**self.t)
        
        # Update biases
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class MLPClassifier:
    """
    Multilayer Perceptron for Human Activity Recognition.
    
    Architecture: 561 -> 256 -> 128 -> 64 -> 6
    Activations: ReLU (hidden) -> Softmax (output)
    """
    
    def __init__(self):
        # Build network
        self.layers = [
            DenseLayer(INPUT_SIZE, 256, 'relu'),
            DenseLayer(256, 128, 'relu'),
            DenseLayer(128, 64, 'relu'),
            DenseLayer(64, NUM_CLASSES, 'softmax')
        ]
        
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = AdamOptimizer(LEARNING_RATE)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray):
        """Backward pass using chain rule"""
        # Start with loss gradient
        dvalues = self.loss_fn.backward(predictions, targets)
        
        # Backpropagate through layers in reverse
        for i, layer in enumerate(reversed(self.layers)):
            dvalues = layer.backward(dvalues)
    
    def update_weights(self):
        """Update all layer weights using optimizer"""
        for i, layer in enumerate(self.layers):
            self.optimizer.update(layer, f'layer_{i}')
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Single training step: forward, loss, backward, update"""
        # Forward pass
        predictions = self.forward(X_batch)
        
        # Calculate loss
        loss = self.loss_fn.forward(predictions, y_batch)
        
        # Backward pass
        self.backward(predictions, y_batch)
        
        # Update weights
        self.update_weights()
        
        return loss
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and probabilities"""
        probabilities = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate accuracy and loss on dataset"""
        predictions, probabilities = self.predict(X)
        y_true = np.argmax(y, axis=1)
        
        accuracy = np.mean(predictions == y_true)
        loss = self.loss_fn.forward(probabilities, y)
        
        return accuracy, loss
    
    def get_weights(self) -> Dict:
        """Export weights and biases for serialization"""
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'layer_{i}_weights'] = layer.weights.tolist()
            # Flatten biases from (1, output_size) to 1D array for JavaScript
            weights[f'layer_{i}_biases'] = layer.biases.flatten().tolist()
        return weights


def load_data(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess UCI HAR dataset.
    
    CRITICAL: Scaling parameters are derived ONLY from training data
    to prevent data leakage.
    """
    print("Loading training data...")
    train_data = np.genfromtxt(train_path, delimiter=',', skip_header=1)
    
    print("Loading test data...")
    test_data = np.genfromtxt(test_path, delimiter=',', skip_header=1)
    
    # Separate features and labels
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].astype(int) - 1  # Convert to 0-indexed
    
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(int) - 1
    
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, y_train, X_test, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Min-max scaling to [-1, 1] range.
    
    CRITICAL: Use ONLY training set statistics to avoid data leakage!
    """
    # Calculate scaling parameters from TRAINING DATA ONLY
    data_min = X_train.min()
    data_max = X_train.max()
    
    print(f"Scaling parameters (from training set only):")
    print(f"  Data min: {data_min:.6f}")
    print(f"  Data max: {data_max:.6f}")
    
    # Apply scaling
    feature_range = (-1, 1)
    data_range = data_max - data_min + EPSILON
    
    X_train_scaled = (X_train - data_min) / data_range * 2 - 1
    X_test_scaled = (X_test - data_min) / data_range * 2 - 1  # Use TRAINING params!
    
    scaling_params = {
        'method': 'minmax',
        'feature_min': feature_range[0],
        'feature_max': feature_range[1],
        'data_min': float(data_min),
        'data_max': float(data_max),
        'epsilon': EPSILON
    }
    
    return X_train_scaled, X_test_scaled, scaling_params


def one_hot_encode(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors"""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create mini-batches for stochastic gradient descent"""
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    batches = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        batches.append((X_batch, y_batch))
    
    return batches


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for model evaluation"""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def train(model: MLPClassifier, X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """
    Train the MLP with early stopping.
    
    Returns training history for visualization.
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Learning rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        # Create shuffled batches
        batches = create_batches(X_train, y_train, BATCH_SIZE)
        
        # Training loop
        epoch_losses = []
        for X_batch, y_batch in batches:
            loss = model.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
        
        # Evaluate on training and validation sets
        train_acc, train_loss = model.evaluate(X_train, y_train)
        val_acc, val_loss = model.evaluate(X_val, y_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.get_weights()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("-" * 60)
    return history, best_weights


def save_model(weights: Dict, scaling_params: Dict, output_path: str):
    """Save trained model weights and scaling parameters to JSON"""
    model_data = {
        'metadata': {
            'architecture': 'MLP',
            'input_size': INPUT_SIZE,
            'hidden_layers': [256, 128, 64],
            'output_size': NUM_CLASSES,
            'activations': ['relu', 'relu', 'relu', 'softmax'],
            'seed': SEED,
            'learning_rate': LEARNING_RATE
        },
        'scaling_params': scaling_params,
        'class_labels': ACTIVITIES,
        'weights': weights
    }
    
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"\nModel saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train HAR MLP model')
    parser.add_argument('--train_path', type=str, default='train.csv',
                        help='Path to training CSV')
    parser.add_argument('--test_path', type=str, default='test.csv',
                        help='Path to test CSV')
    parser.add_argument('--output', type=str, default='../../public/model/weights.json',
                        help='Output path for model weights')
    parser.add_argument('--output_dir', type=str, default='../../public/model',
                        help='Output directory for model artifacts')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HAR MLP Training - Experimentation Cycle")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(args.train_path, args.test_path)
    
    # Scale features (CRITICAL: training params only!)
    X_train_scaled, X_test_scaled, scaling_params = scale_features(X_train, X_test)
    
    # One-hot encode labels
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    # Split training into train/validation
    val_size = int(X_train_scaled.shape[0] * VALIDATION_SPLIT)
    indices = np.random.permutation(X_train_scaled.shape[0])
    
    X_val = X_train_scaled[indices[:val_size]]
    y_val = y_train_oh[indices[:val_size]]
    y_val_labels = y_train[indices[:val_size]]
    X_train_final = X_train_scaled[indices[val_size:]]
    y_train_final = y_train_oh[indices[val_size:]]
    y_train_labels = y_train[indices[val_size:]]
    
    print(f"\nData split:")
    print(f"  Training: {X_train_final.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test_scaled.shape[0]} samples")
    
    # Create and train model
    model = MLPClassifier()
    history, best_weights = train(model, X_train_final, y_train_final, X_val, y_val)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    test_acc, test_loss = model.evaluate(X_test_scaled, y_test_oh)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Confusion matrix
    y_pred, _ = model.predict(X_test_scaled)
    cm = compute_confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print("Predicted ->")
    print(f"{'':12s}", end='')
    for activity in ACTIVITIES:
        print(f"{activity[:6]:>8s}", end='')
    print()
    
    for i, activity in enumerate(ACTIVITIES):
        print(f"{activity[:12]:12s}", end='')
        for j in range(NUM_CLASSES):
            print(f"{cm[i,j]:8d}", end='')
        print()
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, activity in enumerate(ACTIVITIES):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {activity}: {acc:.4f}")
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_model(best_weights, scaling_params, args.output)
    
    # Save confusion matrix
    with open(f"{args.output_dir}/confusion_matrix.json", 'w') as f:
        json.dump(cm.tolist(), f, indent=2)
    
    # Save training history
    with open(f"{args.output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
