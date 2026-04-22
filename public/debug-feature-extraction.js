// Test if feature extraction captures activity differences
// Run in browser console: copy & paste this after model loads

async function debugFeatureExtraction() {
  console.log('=== FEATURE EXTRACTION DEBUG ===\n');

  // Simulate activities
  const activities = ['WALKING', 'WALKING_UPSTAIRS', 'SITTING', 'LAYING'];
  const results = {};

  for (const activity of activities) {
    // Generate 128 samples for this activity
    const accBuffer = [];
    const gyroBuffer = [];

    const baseValues = {
      WALKING: { acc: [0.5, 0.1, -0.95], gyro: [0.1, 0.3, 0.15] },
      WALKING_UPSTAIRS: { acc: [0.4, 0.2, -1.0], gyro: [0.15, 0.4, 0.2] },
      SITTING: { acc: [0.0, 0.0, -0.95], gyro: [0.0, 0.0, 0.0] },
      LAYING: { acc: [0.1, -0.9, 0.0], gyro: [0.0, 0.0, 0.0] },
    };

    const base = baseValues[activity];
    const noise = () => (Math.random() - 0.5) * 0.3;

    for (let i = 0; i < 128; i++) {
      accBuffer.push({
        x: base.acc[0] + noise(),
        y: base.acc[1] + noise(),
        z: base.acc[2] + noise(),
      });
      gyroBuffer.push({
        x: base.gyro[0] + noise() * 0.2,
        y: base.gyro[1] + noise() * 0.2,
        z: base.gyro[2] + noise() * 0.2,
      });
    }

    // Extract features
    const features = window.extractFeatures(accBuffer, gyroBuffer);

    // Calculate feature statistics
    const nonZeroCount = features.filter(f => f !== 0).length;
    const mean = features.reduce((a, b) => a + b, 0) / features.length;
    const variance = features.reduce((a, b) => a + (b - mean) ** 2, 0) / features.length;
    const std = Math.sqrt(variance);

    console.log(`${activity}:`);
    console.log(`  Features: ${features.length}`);
    console.log(`  Non-zero: ${nonZeroCount}`);
    console.log(`  Mean: ${mean.toFixed(4)}, Std: ${std.toFixed(4)}`);
    console.log(
      `  Range: [${Math.min(...features).toFixed(4)}, ${Math.max(...features).toFixed(4)}]`
    );
    console.log(
      `  First 5: [${features
        .slice(0, 5)
        .map(f => f.toFixed(4))
        .join(', ')}]`
    );
    console.log();

    results[activity] = {
      mean,
      std,
      min: Math.min(...features),
      max: Math.max(...features),
      first5: features.slice(0, 5),
    };
  }

  // Check differences
  console.log('=== DIFFERENCES ===');
  const keys = Object.keys(results);
  for (let i = 0; i < keys.length; i++) {
    for (let j = i + 1; j < keys.length; j++) {
      const a = keys[i];
      const b = keys[j];
      const meanDiff = Math.abs(results[a].mean - results[b].mean);
      console.log(`${a} vs ${b}: Mean diff = ${meanDiff.toFixed(6)}`);
    }
  }
}

// Try to access the function
if (typeof window !== 'undefined' && window.extractFeatures) {
  debugFeatureExtraction();
} else {
  console.error('extractFeatures not found in window scope');
}
