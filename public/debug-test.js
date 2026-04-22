/**
 * Simple test to debug feature extraction and model prediction
 * Add this to the live-classifier component temporarily for debugging
 */

const DEBUG_TESTS = {
  testFeatureExtraction() {
    console.log('\n=== TESTING FEATURE EXTRACTION ===');

    // Test 1: Check if features differ between activities
    const activitiesData = {
      WALKING: { acc: [0.5, 0.1, -0.95], gyro: [0.1, 0.3, 0.15] },
      SITTING: { acc: [0.0, 0.0, -0.95], gyro: [0.0, 0.0, 0.0] },
      LAYING: { acc: [0.1, -0.9, 0.0], gyro: [0.0, 0.0, 0.0] },
    };

    const allFeatures = {};

    for (const [activity, baseData] of Object.entries(activitiesData)) {
      const accBuffer = [];
      const gyroBuffer = [];

      // Generate 128 consistent samples (no randomness for testing)
      for (let i = 0; i < 128; i++) {
        accBuffer.push({
          x: baseData.acc[0],
          y: baseData.acc[1],
          z: baseData.acc[2],
        });
        gyroBuffer.push({
          x: baseData.gyro[0],
          y: baseData.gyro[1],
          z: baseData.gyro[2],
        });
      }

      try {
        // Import extractFeatures from inference-engine
        const { extractFeatures } = require('@/lib/inference-engine');
        const features = extractFeatures(accBuffer, gyroBuffer);

        // Check for NaN/Infinity
        const nanCount = features.filter(f => !isFinite(f)).length;
        const zeroCount = features.filter(f => f === 0).length;
        const nonZeroCount = features.length - zeroCount;

        console.log(`${activity}:`);
        console.log(`  Length: ${features.length}`);
        console.log(`  NaN/Infinity: ${nanCount}`);
        console.log(`  Zeros: ${zeroCount} / Non-zeros: ${nonZeroCount}`);

        if (nonZeroCount > 0) {
          const nonZeroValues = features.filter(f => f !== 0);
          console.log(
            `  Range: [${Math.min(...nonZeroValues).toFixed(4)}, ${Math.max(...nonZeroValues).toFixed(4)}]`
          );
          console.log(
            `  Mean (non-zero): ${(nonZeroValues.reduce((a, b) => a + b, 0) / nonZeroValues.length).toFixed(4)}`
          );
        }

        allFeatures[activity] = features;
      } catch (err) {
        console.error(`  ERROR: ${err.message}`);
        allFeatures[activity] = null;
      }
    }

    // Compare features
    console.log('\n=== FEATURE COMPARISON ===');
    const activities = Object.keys(allFeatures);

    if (allFeatures['WALKING'] && allFeatures['SITTING']) {
      const walkingFeats = allFeatures['WALKING'];
      const sittingFeats = allFeatures['SITTING'];

      let diff = 0;
      for (let i = 0; i < walkingFeats.length; i++) {
        diff += Math.abs(walkingFeats[i] - sittingFeats[i]);
      }

      console.log(`Walking vs Sitting: Total absolute difference = ${diff.toFixed(4)}`);
      console.log(`Average difference per feature: ${(diff / walkingFeats.length).toFixed(6)}`);
    }

    return allFeatures;
  },

  testModelPrediction(features) {
    console.log('\n=== TESTING MODEL PREDICTION ===');

    try {
      const { getInferenceEngine } = require('@/lib/inference-engine');
      const engine = getInferenceEngine();

      if (!engine.ready) {
        console.warn('Engine not ready yet');
        return;
      }

      const result = engine.predict(features);

      console.log('Prediction result:');
      console.log(`  Predicted: ${result.multiclass.predicted}`);
      console.log(`  Confidence: ${(result.multiclass.confidence * 100).toFixed(1)}%`);
      console.log(`  Raw logits: [${result.rawLogits.map(l => l.toFixed(3)).join(', ')}]`);
      console.log(`  Probabilities:`);

      for (const activity of [
        'WALKING',
        'WALKING_UPSTAIRS',
        'WALKING_DOWNSTAIRS',
        'SITTING',
        'STANDING',
        'LAYING',
      ]) {
        const prob = result.multiclass.probabilities[activity];
        console.log(`    ${activity}: ${(prob * 100).toFixed(1)}%`);
      }

      return result;
    } catch (err) {
      console.error(`ERROR: ${err.message}`);
      console.error(err.stack);
    }
  },
};

// Run tests
console.log('🔍 Starting debug tests...');
const features = DEBUG_TESTS.testFeatureExtraction();
if (features && features['WALKING']) {
  DEBUG_TESTS.testModelPrediction(features['WALKING']);
}
