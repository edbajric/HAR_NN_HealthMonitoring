// Human Activity Recognition Data Types and Utilities

export type Activity =
  | 'WALKING'
  | 'WALKING_UPSTAIRS'
  | 'WALKING_DOWNSTAIRS'
  | 'SITTING'
  | 'STANDING'
  | 'LAYING';
export type BinaryActivity = 'ACTIVE' | 'SEDENTARY';

export const ACTIVITIES: Activity[] = [
  'WALKING',
  'WALKING_UPSTAIRS',
  'WALKING_DOWNSTAIRS',
  'SITTING',
  'STANDING',
  'LAYING',
];

export const ACTIVITY_LABELS: Record<Activity, string> = {
  WALKING: 'Walking',
  WALKING_UPSTAIRS: 'Upstairs',
  WALKING_DOWNSTAIRS: 'Downstairs',
  SITTING: 'Sitting',
  STANDING: 'Standing',
  LAYING: 'Laying',
};

export const ACTIVITY_COLORS: Record<Activity, string> = {
  WALKING: 'hsl(var(--chart-1))',
  WALKING_UPSTAIRS: 'hsl(var(--chart-2))',
  WALKING_DOWNSTAIRS: 'hsl(var(--chart-3))',
  SITTING: 'hsl(var(--chart-4))',
  STANDING: 'hsl(var(--chart-5))',
  LAYING: 'hsl(var(--chart-6))',
};

export const BINARY_MAPPING: Record<Activity, BinaryActivity> = {
  WALKING: 'ACTIVE',
  WALKING_UPSTAIRS: 'ACTIVE',
  WALKING_DOWNSTAIRS: 'ACTIVE',
  SITTING: 'SEDENTARY',
  STANDING: 'SEDENTARY',
  LAYING: 'SEDENTARY',
};

// Simulated model performance metrics
export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface ConfusionMatrixData {
  actual: Activity;
  predicted: Activity;
  count: number;
}

// Generate realistic confusion matrix data for Neural Network
export function generateNNConfusionMatrix(): ConfusionMatrixData[] {
  const data: ConfusionMatrixData[] = [];

  // Neural Network confusion matrix (higher accuracy)
  const nnMatrix: Record<Activity, Record<Activity, number>> = {
    WALKING: {
      WALKING: 145,
      WALKING_UPSTAIRS: 3,
      WALKING_DOWNSTAIRS: 2,
      SITTING: 0,
      STANDING: 0,
      LAYING: 0,
    },
    WALKING_UPSTAIRS: {
      WALKING: 4,
      WALKING_UPSTAIRS: 138,
      WALKING_DOWNSTAIRS: 8,
      SITTING: 0,
      STANDING: 0,
      LAYING: 0,
    },
    WALKING_DOWNSTAIRS: {
      WALKING: 2,
      WALKING_UPSTAIRS: 6,
      WALKING_DOWNSTAIRS: 132,
      SITTING: 0,
      STANDING: 0,
      LAYING: 0,
    },
    SITTING: {
      WALKING: 0,
      WALKING_UPSTAIRS: 0,
      WALKING_DOWNSTAIRS: 0,
      SITTING: 125,
      STANDING: 15,
      LAYING: 2,
    },
    STANDING: {
      WALKING: 0,
      WALKING_UPSTAIRS: 0,
      WALKING_DOWNSTAIRS: 0,
      SITTING: 12,
      STANDING: 130,
      LAYING: 0,
    },
    LAYING: {
      WALKING: 0,
      WALKING_UPSTAIRS: 0,
      WALKING_DOWNSTAIRS: 0,
      SITTING: 1,
      STANDING: 0,
      LAYING: 149,
    },
  };

  for (const actual of ACTIVITIES) {
    for (const predicted of ACTIVITIES) {
      data.push({
        actual,
        predicted,
        count: nnMatrix[actual][predicted],
      });
    }
  }

  return data;
}

// Generate realistic confusion matrix data for Logistic Regression
export function generateLRConfusionMatrix(): ConfusionMatrixData[] {
  const data: ConfusionMatrixData[] = [];

  // Logistic Regression confusion matrix (lower accuracy)
  const lrMatrix: Record<Activity, Record<Activity, number>> = {
    WALKING: {
      WALKING: 130,
      WALKING_UPSTAIRS: 10,
      WALKING_DOWNSTAIRS: 8,
      SITTING: 1,
      STANDING: 1,
      LAYING: 0,
    },
    WALKING_UPSTAIRS: {
      WALKING: 12,
      WALKING_UPSTAIRS: 118,
      WALKING_DOWNSTAIRS: 18,
      SITTING: 1,
      STANDING: 1,
      LAYING: 0,
    },
    WALKING_DOWNSTAIRS: {
      WALKING: 8,
      WALKING_UPSTAIRS: 15,
      WALKING_DOWNSTAIRS: 115,
      SITTING: 1,
      STANDING: 1,
      LAYING: 0,
    },
    SITTING: {
      WALKING: 1,
      WALKING_UPSTAIRS: 1,
      WALKING_DOWNSTAIRS: 0,
      SITTING: 105,
      STANDING: 30,
      LAYING: 5,
    },
    STANDING: {
      WALKING: 1,
      WALKING_UPSTAIRS: 1,
      WALKING_DOWNSTAIRS: 0,
      SITTING: 25,
      STANDING: 110,
      LAYING: 5,
    },
    LAYING: {
      WALKING: 0,
      WALKING_UPSTAIRS: 0,
      WALKING_DOWNSTAIRS: 0,
      SITTING: 8,
      STANDING: 5,
      LAYING: 137,
    },
  };

  for (const actual of ACTIVITIES) {
    for (const predicted of ACTIVITIES) {
      data.push({
        actual,
        predicted,
        count: lrMatrix[actual][predicted],
      });
    }
  }

  return data;
}

// Calculate metrics from confusion matrix
export function calculateMetrics(confusionData: ConfusionMatrixData[]): ModelMetrics {
  let totalCorrect = 0;
  let total = 0;

  for (const item of confusionData) {
    total += item.count;
    if (item.actual === item.predicted) {
      totalCorrect += item.count;
    }
  }

  const accuracy = totalCorrect / total;

  // Simplified precision/recall calculation
  const precision = accuracy * 0.98;
  const recall = accuracy * 0.97;
  const f1Score = (2 * (precision * recall)) / (precision + recall);

  return {
    accuracy,
    precision,
    recall,
    f1Score,
  };
}

// Generate training loss curve data
export function generateLossData(
  epochs: number = 50
): { epoch: number; trainLoss: number; valLoss: number }[] {
  const data = [];

  for (let i = 1; i <= epochs; i++) {
    const trainLoss = 2.5 * Math.exp(-0.08 * i) + 0.15 + Math.random() * 0.05;
    const valLoss = 2.5 * Math.exp(-0.07 * i) + 0.2 + Math.random() * 0.08;
    data.push({
      epoch: i,
      trainLoss: Math.max(0.1, trainLoss),
      valLoss: Math.max(0.15, valLoss),
    });
  }

  return data;
}

// Generate accuracy curve data
export function generateAccuracyData(
  epochs: number = 50
): { epoch: number; trainAcc: number; valAcc: number }[] {
  const data = [];

  for (let i = 1; i <= epochs; i++) {
    const trainAcc = 0.96 - 0.6 * Math.exp(-0.12 * i) + Math.random() * 0.02;
    const valAcc = 0.94 - 0.6 * Math.exp(-0.1 * i) + Math.random() * 0.03;
    data.push({
      epoch: i,
      trainAcc: Math.min(0.98, trainAcc),
      valAcc: Math.min(0.96, valAcc),
    });
  }

  return data;
}

// Simulated sensor data sample
export interface SensorSample {
  timestamp: number;
  accX: number;
  accY: number;
  accZ: number;
  gyroX: number;
  gyroY: number;
  gyroZ: number;
}

// Generate simulated real-time sensor stream
export function generateSensorSample(activity: Activity): SensorSample {
  const timestamp = Date.now();

  // Base values depend on activity - with more realistic differences
  // Values scaled to represent actual accelerometer/gyro readings
  const baseValues: Record<Activity, { acc: number[]; gyro: number[] }> = {
    WALKING: {
      acc: [0.5, 0.1, -0.95],
      gyro: [0.1, 0.3, 0.15],
    },
    WALKING_UPSTAIRS: {
      acc: [0.4, 0.2, -1.0],
      gyro: [0.15, 0.4, 0.2],
    },
    WALKING_DOWNSTAIRS: {
      acc: [0.45, 0.15, -0.9],
      gyro: [0.12, 0.35, 0.18],
    },
    SITTING: {
      acc: [0.0, 0.0, -0.95],
      gyro: [0.0, 0.0, 0.0],
    },
    STANDING: {
      acc: [0.05, 0.0, -0.98],
      gyro: [0.02, 0.0, 0.02],
    },
    LAYING: {
      acc: [0.1, -0.9, 0.0],
      gyro: [0.0, 0.0, 0.0],
    },
  };

  const base = baseValues[activity];
  const noise = () => (Math.random() - 0.5) * 0.3;

  return {
    timestamp,
    accX: base.acc[0] + noise(),
    accY: base.acc[1] + noise(),
    accZ: base.acc[2] + noise(),
    gyroX: base.gyro[0] + noise() * 0.2,
    gyroY: base.gyro[1] + noise() * 0.2,
    gyroZ: base.gyro[2] + noise() * 0.2,
  };
}

// Dataset statistics
export const DATASET_STATS = {
  totalSamples: 10299,
  trainSamples: 7352,
  testSamples: 2947,
  features: 561,
  subjects: 30,
  samplingRate: 50, // Hz
  windowSize: 2.56, // seconds
  overlap: 50, // percent
  activities: 6,
};

// Per-class distribution
export const CLASS_DISTRIBUTION = [
  { activity: 'Walking', train: 1226, test: 496 },
  { activity: 'Upstairs', train: 1073, test: 471 },
  { activity: 'Downstairs', train: 986, test: 420 },
  { activity: 'Sitting', train: 1286, test: 491 },
  { activity: 'Standing', train: 1374, test: 532 },
  { activity: 'Laying', train: 1407, test: 537 },
];
