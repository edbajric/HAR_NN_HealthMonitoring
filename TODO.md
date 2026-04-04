To ensure your Rational Agent effectively classifies health behavior percepts from the UCI HAR dataset, we must complete the transition from a raw World State to a tractable Search State through rigorous engineering
. The following TODO plan outlines every technical step remaining to build a robust, human-compatible agent that "does the right thing" for the human principal
.
TODO: Human Activity Recognition (HAR) Agent Implementation
Phase 1: Data Abstraction & Percept Processing (Offline Pillar)
[ ] Signal Standardization: In scripts/training/har_training.py, load train.csv and test.csv using NumPy to manage the 50Hz sensory signals
.
[ ] Temporal Windowing: Implement the fixed-width sliding window logic: windows of 2.56 seconds containing 128 readings with a 50% overlap
.
[ ] Gravitational Separation: Implement a low-pass Butterworth filter with a 0.3 Hz cutoff frequency to separate body acceleration from gravitational components
.
[ ] Factored Representation Extraction: Generate the 561-feature vector for each window, calculating time and frequency domain variables (mean, signal energy, entropy)
.
[ ] Normalization Constants: Calculate min/max or mean/std scaling parameters only from the training set to avoid "peeking"
.
[ ] Export Preprocessing Meta: Save these scaling constants to public/model/scaling_params.json for frontend synchronization
.
Phase 2: Neural Network Mastery (Offline Pillar)
[ ] Architectural Definition: Construct a Multilayer Perceptron (MLP) object with sequential Layer_Dense components
.
Input Layer: 561 units
.
Hidden Layers: Minimum two layers using ReLU activations to learn non-linear piecewise approximations
.
Output Layer: 6 units with a Softmax activation to produce a categorical probability distribution
.
[ ] Loss Engine: Implement multi-class cross-entropy loss; include a small epsilon (1e-7) to prevent numerical instability during log calculations
.
[ ] Optimization Cycle: Execute Stochastic Gradient Descent (SGD) or the Adam optimizer using backpropagation and the chain rule
.
[ ] Generalization Safeguards: Implement Early Stopping on a held-out validation set to prevent the agent from memorizing noise (overfitting)
.
[ ] Baseline Construction: Train a simple Logistic Regression model to serve as a performance floor for comparison
.
[ ] Model Serialization: Export the final weights and biases to public/model/weights.json
.
Phase 3: The Inference Bridge & Agent Program (Online Pillar)
[ ] Browser Inference Engine: In lib/inference-engine.ts, implement the mathematical forward pass logic to transform percepts into probability scores using the pre-trained weights
.
[ ] Temporal Buffer in UI: Update live-classifier.tsx to include a data buffer that accumulates 128 sensor readings before triggering inference
.
[ ] Super-class Aggregation Logic: Implement a function to map the six UCI classes into binary super-classes: Active (Walking, Upstairs, Downstairs) and Sedentary (Sitting, Standing, Laying)
.
[ ] Value Alignment Tuning: Bias the classification logic to prioritize "Sedentary" sensitivity, minimizing false negatives to protect user health
.
Phase 4: Actuators & Visualization (Deployment)
[ ] Loss Landscape Visualization: Connect training-charts.tsx to a JSON export of your training history to show the experimentation cycle
.
[ ] Model Comparison Dashboard: Ensure model-comparison.tsx displays Confusion Matrices side-by-side for the MLP and the Logistic Regression baseline
.
[ ] Structural Transparency: Update distribution-chart.tsx to show the full Softmax probability distribution (confidence scores) for each classification action
.
Phase 5: Professional Integrity & Scientific Reporting
[ ] Algorithmic Bias Analysis: Research and document potential biases in the 30-volunteer UCI dataset (demographics, device placement on the waist)
.
[ ] King Midas Problem Assessment: Finalize ethical-considerations.tsx to discuss risks of the agent pursuing classification objectives too literally
.
[ ] Final Report: Document the transition from "black box" algorithms to a human-compatible, interpretable health monitoring system
.

--------------------------------------------------------------------------------
Scientific Note: Do not let your critical thinking skills atrophy by relying on code generators for the mathematical core
. Ensure the Butterworth filter and Softmax implementation are derivationally sound before the final experimentation cycle concludes
.

o successfully deploy a Rational Agent for Human Activity Recognition (HAR), one must navigate a complex landscape of engineering pitfalls. As we move from raw sensor percepts to optimal classification actions, maintaining scientific rigor is paramount to ensure the agent "does the right thing" for human health monitoring
.
Below is a formalized analysis of the most critical mistakes and technical errors to avoid during the development of your Universal Function Approximator
.
1. Violations of the Experimentation Cycle (Data Leakage)
The most common error in supervised learning is failing to strictly isolate information between datasets
.
Hyperparameter Peeking: You must not use the test set to tune hyperparameters such as learning rate or layer size
. If you do, you are manually optimizing the model to the test set, which biases it toward overfitting that specific data
. Use a validation set (held-out data) for all tuning
.
Preprocessing Contamination: Preprocessing, such as scaling or normalization, must be informed only by the training dataset
. A common mistake is querying the entire dataset for min/max values or standard deviations before partitioning
.
Time-Series Correlation: Because smartphone sensor data is sampled at 50Hz, consecutive readings are highly correlated
. Randomly shuffling individual samples into training and test sets leads to "data leakage," where the model memorizes signal noise from adjacent windows
. You must partition data by entire blocks of time or by distinct human subjects to prove the agent's ability to generalize to unseen users
.
2. Optimization and Gradient Failures
The ability of Stochastic Gradient Descent (SGD) to minimize multi-class cross-entropy loss depends on precise parameter control
.
The Learning Rate Dilemma:
Overshooting: A learning rate that is too high causes the model to "jump" around the loss landscape, creating a "flashy wiggle" effect in accuracy charts
.
Gradient Explosion: Critically high rates can lead to values exceeding floating-point limitations (overflow), rendering the network useless
.
Trapped in Local Minima: Conversely, a learning rate that is too low or decays too quickly can trap the model in a local minimum where loss stops decreasing prematurely
.
Failing to Shuffle Training Data: If training data is ordered by label (e.g., all "Walking" then all "Sitting"), the model will cycle through local minima, learning only to predict the most recent label it has seen
.
3. Numerical Instability in Neural Network Mastery
Treating the network as a mathematical object requires accounting for edge-case failures.
The Logarithm Problem: When calculating Categorical Cross-Entropy, a model that is "fully wrong" may assign a confidence of zero to the target label
. Calculating -np.log(0) returns negative infinity, which destroys subsequent gradient calculations
. You must apply clipping (e.g., 1e-7) to confidence scores to prevent this
.
The Dying ReLU Problem: While ReLU activations are standard, their derivative is zero for negative inputs
. If a neuron's weights are adjusted such that it output zero for all samples, it becomes "dead" and non-trainable
.
Improper Initialization: Failing to use careful initialization (like He initialization) can lead to vanishing or exploding gradients, where weight updates either become too small to matter or too large to be stable as they ripple through layers
.
4. Generalization and The "King Midas Problem"
Technical accuracy does not guarantee a safe or useful agent
.
Overfitting (Memorization): With sufficient capacity (hidden units), a network will fit the training data perfectly by memorizing idiosyncrasies rather than understanding input-output dependencies
. You must recognize when validation loss begins to rise while training loss falls
.
The Value Alignment Problem: This is the risk of the agent following a literal objective too strictly, leading to unintended outcomes
. For instance, treating all misclassifications as equal violates the project's performance measure, which requires minimizing false negatives for sedentary behavior to accurately identify health risks
.
Algorithmic Bias: You must account for the fact that your training data (30 volunteers) may be unrepresentative of the broader population, leading to disparate failure rates for different demographics
.
Automation Bias: Practitioners must avoid the assumption that model outputs are "objective" simply because they are mathematical
. Critical thinking skills must not atrophy; the agent should remain a tool that defers to human clinical oversight
.
By strictly adhering to the Experimentation Cycle and maintaining professional integrity regarding these risks, you will ensure the agent provides an interpretable and beneficial health monitoring system
.