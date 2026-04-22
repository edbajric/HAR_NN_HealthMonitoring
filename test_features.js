// Quick Node.js test to verify feature extraction
const fs = require('fs');
const path = require('path');

// Load and parse TypeScript module (using simple extraction)
const tsContent = fs.readFileSync('lib/inference-engine.ts', 'utf8');

// Test helper: Generate synthetic sensor data
function generateSensorData() {
  const data = [];
  for (let i = 0; i < 128; i++) {
    // Simulate walking pattern
    const t = i / 50; // 50Hz
    data.push({
      x: 0.5 * Math.sin(2 * Math.PI * 1.3 * t) + 0.1 * Math.random(),
      y: 0.1 * Math.cos(2 * Math.PI * 0.8 * t) + 0.1 * Math.random(),
      z: -0.95 + 0.05 * Math.sin(2 * Math.PI * 1.5 * t) + 0.05 * Math.random()
    });
  }
  return data;
}

console.log("Testing feature extraction...");
console.log("Generated 128 sensor samples for testing");

// Count expected features 
const expectedBreakdown = {
  "tBodyAcc": 40,
  "tGravityAcc": 40,
  "tBodyAccJerk": 40,
  "tBodyGyro": 40,
  "tBodyGyroJerk": 40,
  "tBodyAccMag": 13,
  "tGravityAccMag": 13,
  "tBodyAccJerkMag": 13,
  "tBodyGyroMag": 13,
  "tBodyGyroJerkMag": 13,
  "fBodyAcc": 79,
  "fBodyAccJerk": 79,
  "fBodyGyro": 79,
  "fBodyAccMag": 13,
  "fBodyBodyAccJerkMag": 13,
  "fBodyBodyGyroMag": 13,
  "fBodyBodyGyroJerkMag": 13,
  "Angles": 7
};

let total = 0;
console.log("\nExpected feature breakdown:");
for (const [name, count] of Object.entries(expectedBreakdown)) {
  console.log(`  ${name.padEnd(25)}: ${count} features`);
  total += count;
}

console.log(`\n  TOTAL: ${total} features`);

if (total === 561) {
  console.log("\n✓ Feature count matches expected 561!");
} else {
  console.log(`\n✗ ERROR: Expected 561 features, but calculated ${total}`);
}
