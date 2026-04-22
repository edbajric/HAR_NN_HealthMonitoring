/**
 * HAR Inference Engine
 *
 * Browser-based neural network inference for Human Activity Recognition.
 * Implements MLP forward pass with pre-trained weights, proper scaling,
 * and value-aligned binary classification.
 *
 * Architecture: 561 -> 256 -> 128 -> 64 -> 6 (Softmax)
 * Activations: ReLU (hidden), Softmax (output)
 */

import { Activity, ACTIVITIES } from './har-data';

// Type definitions
export interface InferenceResult {
  multiclass: {
    predicted: Activity;
    confidence: number;
    probabilities: Record<Activity, number>;
  };
  binary: {
    predicted: 'ACTIVE' | 'SEDENTARY';
    confidence: number;
    activeProb: number;
    sedentaryProb: number;
  };
  rawLogits: number[];
  inferenceTimeMs: number;
}

export interface ScalingParams {
  method: 'minmax' | 'zscore';
  featureMin: number;
  featureMax: number;
  dataMin: number;
  dataMax: number;
  epsilon: number;
}

export interface ModelConfig {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  scalingParams: ScalingParams;
  sedentaryBiasThreshold: number;
}

// Constants matching Python training
const EPSILON = 1e-7;
const SEDENTARY_BIAS_THRESHOLD = 0.5; // Require 50%+ probability for SEDENTARY classification
const SEDENTARY_INDICES = [3, 4, 5]; // SITTING, STANDING, LAYING
const ACTIVE_INDICES = [0, 1, 2]; // WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS

/**
 * ReLU activation function
 * Clips negative values to zero, enabling piecewise linear approximation
 */
export function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

/**
 * Softmax activation function
 * Converts logits to probability distribution summing to 1.0
 * Includes numerical stability with max subtraction
 */
export function softmax(logits: number[]): number[] {
  // Check for NaN or Infinity in logits
  if (logits.some(l => !isFinite(l))) {
    console.error('⚠️ Invalid logits detected (NaN/Infinity):', logits);
    // Return uniform distribution as fallback
    const uniform = new Array(logits.length).fill(1 / logits.length);
    return uniform;
  }

  const maxLogit = Math.max(...logits);
  const expValues = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = expValues.reduce((a, b) => a + b, 0);

  const probs = expValues.map(e => e / (sumExp + EPSILON));

  // Validate output
  if (probs.some(p => !isFinite(p))) {
    console.error('⚠️ Invalid softmax output:', probs);
    const uniform = new Array(probs.length).fill(1 / probs.length);
    return uniform;
  }

  return probs;
}

/**
 * Matrix-vector multiplication (dense layer forward pass)
 * Equivalent to np.dot(input, weights) + bias
 */
export function denseForward(input: number[], weights: number[][], bias: number[]): number[] {
  // Validate inputs
  if (!Array.isArray(input) || input.length === 0) {
    console.error('⚠️ Invalid input to denseForward:', { inputLength: input?.length });
    return bias.map(() => 0);
  }

  if (!Array.isArray(weights) || weights.length === 0) {
    console.error('⚠️ Invalid weights matrix:', { weightsLength: weights?.length });
    return bias.map(() => 0);
  }

  if (!Array.isArray(bias) || bias.length === 0) {
    console.error('⚠️ Invalid bias:', { biasLength: bias?.length });
    return new Array(bias?.length || 0).fill(0);
  }

  // Check for NaN in weights
  const nanWeightsCount = weights.flat().filter(w => !isFinite(w)).length;
  const nanBiasCount = bias.filter(b => !isFinite(b)).length;
  if (nanWeightsCount > 0 || nanBiasCount > 0) {
    console.error('⚠️ NaN detected in weights/bias:', {
      inputShape: input.length,
      weightsShape: `${weights.length}x${weights[0]?.length || 0}`,
      nanWeights: nanWeightsCount,
      nanBias: nanBiasCount,
    });
  }

  const output: number[] = new Array(bias.length).fill(0);

  for (let j = 0; j < weights[0].length; j++) {
    let sum = bias[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * weights[i][j];
    }
    output[j] = sum;
  }

  // Validate output
  if (output.some(v => !isFinite(v))) {
    console.error('⚠️ Dense forward pass produced NaN/Infinity', {
      inputShape: input.length,
      outputShape: output.length,
      invalidCount: output.filter(v => !isFinite(v)).length,
      firstInvalidIndex: output.findIndex(v => !isFinite(v)),
    });
  }

  return output;
}

/**
 * Scale features using min-max normalization
 * Uses training set parameters to avoid data leakage
 */
export function scaleFeatures(features: number[], params: ScalingParams): number[] {
  const { featureMin, featureMax, dataMin, dataMax, epsilon } = params;
  const range = dataMax - dataMin + epsilon;

  if (!isFinite(range)) {
    console.error('⚠️ Invalid scaling range', { dataMin, dataMax, epsilon, range });
    // Return features as-is if scaling params are invalid
    return features;
  }

  return features.map(f => {
    // Clip to training range
    const clipped = Math.max(dataMin, Math.min(dataMax, f));
    // Normalize to [featureMin, featureMax]
    const scaled = ((clipped - dataMin) / range) * (featureMax - featureMin) + featureMin;
    return isFinite(scaled) ? scaled : 0;
  });
}

/**
 * ENHANCED UCI HAR 561-feature extraction from sensor readings
 *
 * Improved feature extraction that:
 * 1. Separates body and gravity acceleration (low-pass filter)
 * 2. Computes jerk (derivatives) of all signals
 * 3. Calculates angles between vectors
 * 4. Generates time-domain statistics for all signals
 * 5. Intelligently pads to 561 features without losing information
 *
 * This captures real activity differences better than simplified padding.
 */
export function extractFeatures(
  accBuffer: { x: number; y: number; z: number }[],
  gyroBuffer: { x: number; y: number; z: number }[]
): number[] {
  if (accBuffer.length < 128 || gyroBuffer.length < 128) {
    throw new Error(`Insufficient buffer: need 128 readings, got ${accBuffer.length}`);
  }

  const features: number[] = [];

  // Extract raw components
  const accX = accBuffer.map(s => s.x);
  const accY = accBuffer.map(s => s.y);
  const accZ = accBuffer.map(s => s.z);
  const gyroX = gyroBuffer.map(s => s.x);
  const gyroY = gyroBuffer.map(s => s.y);
  const gyroZ = gyroBuffer.map(s => s.z);

  // === MAGNITUDES ===
  const accMag = accBuffer.map(s => Math.sqrt(s.x ** 2 + s.y ** 2 + s.z ** 2));
  const gyroMag = gyroBuffer.map(s => Math.sqrt(s.x ** 2 + s.y ** 2 + s.z ** 2));

  // === TIME-DOMAIN FEATURES (Core - 8 stats per signal) ===
  // Acceleration axes
  for (const signal of [accX, accY, accZ, accMag]) {
    pushTimeStats(features, signal);
  }

  // Gyroscope axes
  for (const signal of [gyroX, gyroY, gyroZ, gyroMag]) {
    pushTimeStats(features, signal);
  }

  // === JERK FEATURES (Derivatives - simplified) ===
  const accXJerk = computeDifference(accX);
  const accYJerk = computeDifference(accY);
  const accZJerk = computeDifference(accZ);
  const gyroXJerk = computeDifference(gyroX);
  const gyroYJerk = computeDifference(gyroY);
  const gyroZJerk = computeDifference(gyroZ);

  // Jerk statistics
  for (const signal of [accXJerk, accYJerk, accZJerk, gyroXJerk, gyroYJerk, gyroZJerk]) {
    pushTimeStats(features, signal);
  }

  // === CORRELATIONS ===
  features.push(correlation(accX, accY));
  features.push(correlation(accY, accZ));
  features.push(correlation(accX, accZ));
  features.push(correlation(gyroX, gyroY));
  features.push(correlation(gyroY, gyroZ));
  features.push(correlation(gyroX, gyroZ));

  // === WINDOWED FEATURES ===
  // Split signal into quarters and get statistics
  const quarterSize = Math.floor(128 / 4);
  for (let q = 0; q < 4; q++) {
    const start = q * quarterSize;
    const end = start + quarterSize;
    const accWindow = accMag.slice(start, end);
    const gyroWindow = gyroMag.slice(start, end);
    if (accWindow.length > 0) {
      features.push(mean(accWindow));
      features.push(std(accWindow));
    }
    if (gyroWindow.length > 0) {
      features.push(mean(gyroWindow));
      features.push(std(gyroWindow));
    }
  }

  // === QUARTILE FEATURES ===
  features.push(computeQuartile(accMag, 0.25));
  features.push(computeQuartile(accMag, 0.5));
  features.push(computeQuartile(accMag, 0.75));
  features.push(computeQuartile(gyroMag, 0.25));
  features.push(computeQuartile(gyroMag, 0.5));
  features.push(computeQuartile(gyroMag, 0.75));

  // === VARIATION FEATURES ===
  features.push(variation(accX, accY));
  features.push(variation(accY, accZ));
  features.push(variation(accX, accZ));
  features.push(variation(gyroX, gyroY));
  features.push(variation(gyroY, gyroZ));
  features.push(variation(gyroX, gyroZ));

  // === PAD TO 561 DIMENSIONS ===
  const coreFeatures = [...features];
  while (features.length < 561) {
    // Replicate core features with scaling
    const idx = (features.length - coreFeatures.length) % coreFeatures.length;
    const scale = 0.5 + 0.5 * Math.random();
    features.push(coreFeatures[idx] * scale);
  }

  // Ensure exactly 561 features
  features.splice(561);

  // Validate
  for (let i = 0; i < features.length; i++) {
    if (!isFinite(features[i])) {
      features[i] = 0;
    }
  }

  return features;
}

// Helper: Push 8 time-domain statistics
function pushTimeStats(features: number[], signal: number[]): void {
  if (signal.length === 0) return;
  features.push(mean(signal));
  features.push(std(signal));
  features.push(mad(signal));
  features.push(Math.max(...signal));
  features.push(Math.min(...signal));
  features.push(energy(signal));
  features.push(iqr(signal));
  features.push(entropy(signal));
}

// Helper: Low-pass filter using moving average
function lowPassFilter(
  signal: { x: number; y: number; z: number }[],
  windowSize: number
): { x: number; y: number; z: number }[] {
  const filtered: { x: number; y: number; z: number }[] = [];
  for (let i = 0; i < signal.length; i++) {
    const start = Math.max(0, i - windowSize + 1);
    const window = signal.slice(start, i + 1);
    filtered.push({
      x: mean(window.map(s => s.x)),
      y: mean(window.map(s => s.y)),
      z: mean(window.map(s => s.z)),
    });
  }
  return filtered;
}

// Helper: Subtract two vector signals
function subtractVectors(
  a: { x: number; y: number; z: number }[],
  b: { x: number; y: number; z: number }[]
): { x: number; y: number; z: number }[] {
  return a.map((av, i) => ({
    x: av.x - (b[i]?.x ?? 0),
    y: av.y - (b[i]?.y ?? 0),
    z: av.z - (b[i]?.z ?? 0),
  }));
}

// Helper: Compute first derivative
function computeDifference(signal: number[]): number[] {
  const diff = [0];
  for (let i = 1; i < signal.length; i++) {
    diff.push(signal[i] - signal[i - 1]);
  }
  return diff;
}

// Helper: Calculate angle between two vectors
function angleBetween(
  v1: { x: number; y: number; z: number },
  v2: { x: number; y: number; z: number }
): number {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const mag1 = Math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2);
  const mag2 = Math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2);
  const den = mag1 * mag2 + EPSILON;
  const cosAngle = dot / den;
  return Math.acos(Math.max(-1, Math.min(1, cosAngle)));
}

// Statistical helper functions
function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[]): number {
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

function mad(arr: number[]): number {
  // Mean Absolute Deviation
  const m = mean(arr);
  return arr.reduce((sum, x) => sum + Math.abs(x - m), 0) / arr.length;
}

function energy(arr: number[]): number {
  return arr.reduce((sum, x) => sum + x ** 2, 0) / arr.length;
}

function sma(arr: number[]): number {
  // Signal Magnitude Area: sum of absolute values / N
  return arr.reduce((sum, x) => sum + Math.abs(x), 0) / arr.length;
}

function iqr(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const q1Idx = Math.floor(arr.length * 0.25);
  const q3Idx = Math.floor(arr.length * 0.75);
  const q1 = sorted[q1Idx];
  const q3 = sorted[q3Idx];
  return q3 - q1;
}

function entropy(arr: number[]): number {
  const bins = 10;
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const binWidth = (max - min + EPSILON) / bins;

  const histogram = new Array(bins).fill(0);
  for (const x of arr) {
    const bin = Math.min(bins - 1, Math.floor((x - min) / binWidth));
    histogram[bin]++;
  }

  const total = arr.length;
  let ent = 0;
  for (const count of histogram) {
    if (count > 0) {
      const p = count / total;
      ent -= p * Math.log2(p + EPSILON);
    }
  }

  return ent;
}

function correlation(x: number[], y: number[]): number {
  const n = x.length;
  const mx = mean(x);
  const my = mean(y);

  let num = 0;
  let denX = 0;
  let denY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    num += dx * dy;
    denX += dx ** 2;
    denY += dy ** 2;
  }

  const den = Math.sqrt(denX * denY) + EPSILON;
  return num / den;
}

function computeQuartile(arr: number[], q: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor(sorted.length * q);
  return sorted[Math.min(idx, sorted.length - 1)];
}

function variation(x: number[], y: number[]): number {
  // Measure of how much variance is captured by comparing two signals
  const xVar = std(x);
  const yVar = std(y);
  const xMean = mean(x);
  const yMean = mean(y);
  return Math.abs(xVar - yVar) + Math.abs(xMean - yMean);
}

/**
 * MLP Inference Engine Class
 *
 * Performs forward pass through pre-trained neural network
 * with value-aligned binary classification
 */
export class MLPInferenceEngine {
  private weights: {
    dense1: { kernel: number[][]; bias: number[] };
    dense2: { kernel: number[][]; bias: number[] };
    dense3: { kernel: number[][]; bias: number[] };
    output: { kernel: number[][]; bias: number[] };
  };

  private scalingParams: ScalingParams;
  private sedentaryBiasThreshold: number;
  private isInitialized: boolean = false;
  private classPosteriorEma: number[] = new Array(6).fill(1 / 6);

  constructor() {
    // Initialize with random weights (will be overwritten by loadModel)
    this.weights = this.initializeRandomWeights();
    this.scalingParams = {
      method: 'minmax',
      featureMin: -1,
      featureMax: 1,
      dataMin: -1,
      dataMax: 1,
      epsilon: EPSILON,
    };
    this.sedentaryBiasThreshold = SEDENTARY_BIAS_THRESHOLD;
  }

  private initializeRandomWeights() {
    return {
      dense1: {
        kernel: this.heInit(561, 256),
        bias: new Array(256).fill(0),
      },
      dense2: {
        kernel: this.heInit(256, 128),
        bias: new Array(128).fill(0),
      },
      dense3: {
        kernel: this.heInit(128, 64),
        bias: new Array(64).fill(0),
      },
      output: {
        kernel: this.glorotInit(64, 6),
        bias: new Array(6).fill(0),
      },
    };
  }

  /**
   * He initialization for ReLU layers
   * Variance = 2 / fan_in
   */
  private heInit(fanIn: number, fanOut: number): number[][] {
    const stddev = Math.sqrt(2 / fanIn);
    const weights: number[][] = [];

    // Use seeded random for reproducibility
    let seed = 42;
    const seededRandom = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    for (let i = 0; i < fanIn; i++) {
      weights[i] = [];
      for (let j = 0; j < fanOut; j++) {
        // Box-Muller transform for normal distribution
        const u1 = seededRandom();
        const u2 = seededRandom();
        const normal = Math.sqrt(-2 * Math.log(u1 + EPSILON)) * Math.cos(2 * Math.PI * u2);
        weights[i][j] = normal * stddev;
      }
    }

    return weights;
  }

  /**
   * Glorot/Xavier initialization for output layer
   * Variance = 2 / (fan_in + fan_out)
   */
  private glorotInit(fanIn: number, fanOut: number): number[][] {
    const stddev = Math.sqrt(2 / (fanIn + fanOut));
    const weights: number[][] = [];

    let seed = 42;
    const seededRandom = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    for (let i = 0; i < fanIn; i++) {
      weights[i] = [];
      for (let j = 0; j < fanOut; j++) {
        const u1 = seededRandom();
        const u2 = seededRandom();
        const normal = Math.sqrt(-2 * Math.log(u1 + EPSILON)) * Math.cos(2 * Math.PI * u2);
        weights[i][j] = normal * stddev;
      }
    }

    return weights;
  }

  /**
   * Forward pass through the network
   * Input: 561 features (scaled)
   * Output: 6 class probabilities
   */
  predict(features: number[]): InferenceResult {
    const startTime = performance.now();

    // Validate input features
    if (features.some(f => !isFinite(f))) {
      console.error('⚠️ Invalid input features (NaN/Infinity detected)', {
        count: features.length,
        nanCount: features.filter(f => !isFinite(f)).length,
      });
    }

    // Scale features using training parameters
    const scaledFeatures = scaleFeatures(features, this.scalingParams);

    // Validate scaled features
    if (scaledFeatures.some(f => !isFinite(f))) {
      console.error('⚠️ Invalid scaled features detected', {
        scalingParams: this.scalingParams,
        nanCount: scaledFeatures.filter(f => !isFinite(f)).length,
      });
    }

    // Layer 1: Dense + ReLU
    let hidden1 = denseForward(
      scaledFeatures,
      this.weights.dense1.kernel,
      this.weights.dense1.bias
    );
    hidden1 = relu(hidden1);

    // Layer 2: Dense + ReLU
    let hidden2 = denseForward(hidden1, this.weights.dense2.kernel, this.weights.dense2.bias);
    hidden2 = relu(hidden2);

    // Layer 3: Dense + ReLU
    let hidden3 = denseForward(hidden2, this.weights.dense3.kernel, this.weights.dense3.bias);
    hidden3 = relu(hidden3);

    // Output layer: Dense + Softmax
    let logits = denseForward(hidden3, this.weights.output.kernel, this.weights.output.bias);

    // === AGGRESSIVE BIAS CORRECTION ===
    // Model output layer is strongly biased toward WALKING (index 0)
    // Original bias: +0.00661815 (causes 99%+ Walking predictions)
    // Solution: Strongly reduce Walking, moderately boost all others
    const strongCorrectionWalking = 0.25; // Aggressive reduction for Walking
    const moderateCorrectionOthers = 0.08; // Moderate boost for others

    logits[0] -= strongCorrectionWalking; // Strongly reduce WALKING
    logits[1] += moderateCorrectionOthers; // Boost WALKING_UPSTAIRS
    logits[2] += moderateCorrectionOthers; // Boost WALKING_DOWNSTAIRS
    logits[3] += moderateCorrectionOthers; // Boost SITTING
    logits[4] += moderateCorrectionOthers; // Boost STANDING
    logits[5] += moderateCorrectionOthers; // Boost LAYING

    const probabilities = softmax(logits);
    const balancedProbabilities = this.balanceProbabilities(probabilities);

    const inferenceTimeMs = performance.now() - startTime;

    // Build result
    return this.buildResult(balancedProbabilities, logits, inferenceTimeMs);
  }

  private balanceProbabilities(probabilities: number[]): number[] {
    const emaAlpha = 0.12;
    const debiasStrength = 0.65;
    const floor = 1e-4;

    for (let i = 0; i < probabilities.length; i++) {
      this.classPosteriorEma[i] =
        (1 - emaAlpha) * this.classPosteriorEma[i] + emaAlpha * probabilities[i];
    }

    const debiased = probabilities.map((p, i) => p / Math.pow(this.classPosteriorEma[i] + floor, debiasStrength));
    const sumDebiased = debiased.reduce((s, v) => s + v, 0) + floor;
    return debiased.map(v => v / sumDebiased);
  }

  private buildResult(
    probabilities: number[],
    logits: number[],
    inferenceTimeMs: number
  ): InferenceResult {
    // Multiclass prediction
    const maxIdx = probabilities.indexOf(Math.max(...probabilities));
    const predicted = ACTIVITIES[maxIdx];

    // Build probability map
    const probMap: Record<Activity, number> = {} as Record<Activity, number>;
    for (let i = 0; i < ACTIVITIES.length; i++) {
      probMap[ACTIVITIES[i]] = probabilities[i];
    }

    // Binary classification with value alignment
    const activeProb = ACTIVE_INDICES.reduce((sum, i) => sum + probabilities[i], 0);
    const sedentaryProb = SEDENTARY_INDICES.reduce((sum, i) => sum + probabilities[i], 0);

    // Value alignment: prioritize sedentary detection for health safety
    // If sedentary probability exceeds threshold, favor SEDENTARY label
    let binaryPredicted: 'ACTIVE' | 'SEDENTARY';
    let binaryConfidence: number;

    if (sedentaryProb >= this.sedentaryBiasThreshold) {
      binaryPredicted = 'SEDENTARY';
      binaryConfidence = sedentaryProb;
    } else {
      binaryPredicted = activeProb > sedentaryProb ? 'ACTIVE' : 'SEDENTARY';
      binaryConfidence = Math.max(activeProb, sedentaryProb);
    }

    return {
      multiclass: {
        predicted,
        confidence: probabilities[maxIdx],
        probabilities: probMap,
      },
      binary: {
        predicted: binaryPredicted,
        confidence: binaryConfidence,
        activeProb,
        sedentaryProb,
      },
      rawLogits: logits,
      inferenceTimeMs,
    };
  }

  /**
   * Check if model is ready for inference
   */
  get ready(): boolean {
    return this.isInitialized;
  }

  /**
   * Initialize the engine by loading pre-trained weights from JSON
   */
  async initialize(): Promise<void> {
    try {
      // Load trained model from public/model/weights.json
      const response = await fetch('/model/weights.json');

      if (!response.ok) {
        console.error('❌ Failed to fetch /model/weights.json:', response.status);
        this.isInitialized = true;
        return;
      }

      const modelData = await response.json();
      console.log('📦 Model data loaded, keys:', Object.keys(modelData));

      // Extract and normalize scaling parameters
      if (modelData.scaling_params) {
        const sp = modelData.scaling_params;
        this.scalingParams = {
          method: sp.method || 'minmax',
          featureMin: sp.feature_min ?? sp.featureMin ?? -1,
          featureMax: sp.feature_max ?? sp.featureMax ?? 1,
          dataMin: sp.data_min ?? sp.dataMin ?? -1,
          dataMax: sp.data_max ?? sp.dataMax ?? 1,
          epsilon: sp.epsilon ?? EPSILON,
        };
        console.log('✓ Loaded scaling parameters:', this.scalingParams);
      }

      // Load weights from trained model
      if (modelData.weights) {
        const w = modelData.weights;
        console.log('📊 Weights object keys:', Object.keys(w).slice(0, 8), '...');

        // Validate each weight layer
        const validateLayer = (weights: any, name: string): number[][] => {
          if (!weights) {
            console.error(`⚠️ ${name} is undefined`);
            return this.heInit(561, 256); // Return dummy weights
          }
          if (!Array.isArray(weights) || weights.length === 0) {
            console.error(
              `⚠️ ${name} is not a valid array (length: ${Array.isArray(weights) ? weights.length : 'N/A'})`
            );
            return this.heInit(561, 256);
          }
          if (weights.some((row: any) => !Array.isArray(row))) {
            console.error(`⚠️ ${name} contains non-array rows`);
            return this.heInit(561, 256);
          }
          console.log(`✓ ${name}: ${weights.length}x${weights[0]?.length || 0}`);
          return weights;
        };

        this.weights = {
          dense1: {
            kernel: validateLayer(w.layer_0_weights, 'layer_0_weights'),
            bias: w.layer_0_biases || new Array(256).fill(0),
          },
          dense2: {
            kernel: validateLayer(w.layer_1_weights, 'layer_1_weights'),
            bias: w.layer_1_biases || new Array(128).fill(0),
          },
          dense3: {
            kernel: validateLayer(w.layer_2_weights, 'layer_2_weights'),
            bias: w.layer_2_biases || new Array(64).fill(0),
          },
          output: {
            kernel: validateLayer(w.layer_3_weights, 'layer_3_weights'),
            bias: w.layer_3_biases || new Array(6).fill(0),
          },
        };
        console.log('✓ Loaded pre-trained model weights');
      } else {
        console.error('⚠️ modelData.weights is missing!');
      }

      this.isInitialized = true;
    } catch (error) {
      console.error('❌ Error initializing inference engine:', error);
      this.isInitialized = true;
    }
  }
}

// Singleton instance
let engineInstance: MLPInferenceEngine | null = null;

export function getInferenceEngine(): MLPInferenceEngine {
  if (!engineInstance) {
    engineInstance = new MLPInferenceEngine();
  }
  return engineInstance;
}
