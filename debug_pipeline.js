/**
 * COMPREHENSIVE DEBUG: Test entire pipeline
 * This tests:
 * 1. Sensor generation differs by activity
 * 2. Feature extraction differs by activity
 * 3. Model prediction differs by activity
 */

const fs = require('fs');
const path = require('path');

// Load model
const modelPath = path.join(__dirname, 'public/model/weights.json');
const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));

console.log('='.repeat(80));
console.log('COMPREHENSIVE PIPELINE DEBUG');
console.log('='.repeat(80));

// Base sensor data for each activity
const baseValues = {
  WALKING: { acc: [0.5, 0.1, -0.95], gyro: [0.1, 0.3, 0.15] },
  WALKING_UPSTAIRS: { acc: [0.4, 0.2, -1.0], gyro: [0.15, 0.4, 0.2] },
  WALKING_DOWNSTAIRS: { acc: [0.45, 0.15, -0.9], gyro: [0.12, 0.35, 0.18] },
  SITTING: { acc: [0.0, 0.0, -0.95], gyro: [0.0, 0.0, 0.0] },
  STANDING: { acc: [0.05, 0.0, -0.98], gyro: [0.02, 0.0, 0.02] },
  LAYING: { acc: [0.1, -0.9, 0.0], gyro: [0.0, 0.0, 0.0] },
};

// Test 1: Check sensor values
console.log('\n1. SENSOR DATA GENERATION');
console.log('-'.repeat(80));
for (const [activity, baseData] of Object.entries(baseValues)) {
  console.log(`\n${activity}:`);
  console.log(`  Acc: [${baseData.acc.map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`  Gyro: [${baseData.gyro.map(v => v.toFixed(3)).join(', ')}]`);

  // Compute magnitude
  const accMag = Math.sqrt(baseData.acc[0] ** 2 + baseData.acc[1] ** 2 + baseData.acc[2] ** 2);
  const gyroMag = Math.sqrt(baseData.gyro[0] ** 2 + baseData.gyro[1] ** 2 + baseData.gyro[2] ** 2);
  console.log(`  AccMag: ${accMag.toFixed(4)}, GyroMag: ${gyroMag.toFixed(4)}`);
}

// Test 2: Check if different sensors create different mean values
console.log('\n\n2. MEAN FEATURE VALUES');
console.log('-'.repeat(80));
const means = {};
for (const [activity, baseData] of Object.entries(baseValues)) {
  // Create 128 samples of this activity (with NO noise for purity)
  const samples = [];
  for (let i = 0; i < 128; i++) {
    samples.push({
      acc: baseData.acc,
      gyro: baseData.gyro,
    });
  }

  // Compute key statistics
  const accX = samples.map(s => s.acc[0]);
  const accY = samples.map(s => s.acc[1]);
  const accZ = samples.map(s => s.acc[2]);
  const gyroX = samples.map(s => s.gyro[0]);
  const gyroY = samples.map(s => s.gyro[1]);
  const gyroZ = samples.map(s => s.gyro[2]);

  const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

  const means_for_activity = {
    accX: mean(accX),
    accY: mean(accY),
    accZ: mean(accZ),
    gyroX: mean(gyroX),
    gyroY: mean(gyroY),
    gyroZ: mean(gyroZ),
  };

  means[activity] = means_for_activity;
  console.log(`\n${activity}:`);
  console.log(`  accX mean: ${means_for_activity.accX.toFixed(4)}`);
  console.log(`  accY mean: ${means_for_activity.accY.toFixed(4)}`);
  console.log(`  accZ mean: ${means_for_activity.accZ.toFixed(4)}`);
  console.log(`  gyroX mean: ${means_for_activity.gyroX.toFixed(4)}`);
  console.log(`  gyroY mean: ${means_for_activity.gyroY.toFixed(4)}`);
  console.log(`  gyroZ mean: ${means_for_activity.gyroZ.toFixed(4)}`);
}

// Compare key differences
console.log('\n\n3. FEATURE DIFFERENCES BETWEEN ACTIVITIES');
console.log('-'.repeat(80));

const activities = Object.keys(means);
for (let i = 0; i < activities.length; i++) {
  for (let j = i + 1; j < activities.length; j++) {
    const act1 = activities[i];
    const act2 = activities[j];
    const mean1 = means[act1];
    const mean2 = means[act2];

    // Calculate total difference in key channels
    let diff = 0;
    diff += Math.abs(mean1.accX - mean2.accX);
    diff += Math.abs(mean1.accY - mean2.accY);
    diff += Math.abs(mean1.accZ - mean2.accZ);
    diff += Math.abs(mean1.gyroX - mean2.gyroX);
    diff += Math.abs(mean1.gyroY - mean2.gyroY);
    diff += Math.abs(mean1.gyroZ - mean2.gyroZ);

    console.log(`${act1} vs ${act2}: Total diff = ${diff.toFixed(4)}`);
  }
}

// Test 4: Check model output layer
console.log('\n\n4. MODEL OUTPUT LAYER ANALYSIS');
console.log('-'.repeat(80));
const outputBias = model.weights.layer_3_biases;
console.log('Output biases for each class:');
for (let i = 0; i < outputBias.length; i++) {
  const className = model.class_labels[i];
  console.log(`  [${i}] ${className.padEnd(20)}: ${outputBias[i].toFixed(6)}`);
}

// Check if Walking is "special"
const avgBias = outputBias.reduce((a, b) => a + b, 0) / outputBias.length;
const walkingBias = outputBias[0];
console.log(`\nAverage bias: ${avgBias.toFixed(6)}`);
console.log(`Walking bias: ${walkingBias.toFixed(6)}`);
console.log(`Walking advantage: ${(walkingBias - avgBias).toFixed(6)}`);

// Test 5: Check scaling parameters
console.log('\n\n5. SCALING PARAMETERS');
console.log('-'.repeat(80));
const sp = model.scaling_params;
console.log(`Method: ${sp.method}`);
console.log(`Data range used for training: [${sp.data_min}, ${sp.data_max}]`);
console.log(`Feature range after scaling: [${sp.feature_min}, ${sp.feature_max}]`);
console.log(`Epsilon: ${sp.epsilon}`);

// Check if sensor values are within training range
console.log('\nChecking if current sensor values are within training range:');
let outOfRange = false;
for (const [activity, baseData] of Object.entries(baseValues)) {
  const allValues = [...baseData.acc, ...baseData.gyro];
  const outOfRangeCount = allValues.filter(v => v < sp.data_min || v > sp.data_max).length;
  if (outOfRangeCount > 0) {
    console.log(
      `  ⚠️  ${activity}: ${outOfRangeCount} values out of range ${sp.data_min}-${sp.data_max}`
    );
    outOfRange = true;
  }
}
if (!outOfRange) {
  console.log('  ✓ All sensor values within training range');
}

console.log('\n' + '='.repeat(80));
console.log('END DEBUG');
console.log('='.repeat(80));
