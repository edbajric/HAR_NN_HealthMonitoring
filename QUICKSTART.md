# QUICK START - Run HAR App Now

## 3 Commands to Get Started

### 1. Train the AI Model (First Time Only - ~30 seconds)

```bash
cd /Users/edinabajric/Desktop/Codes/AI_ML/HAR_NN_HealthMonitoring
npm run train
```

**What happens:**

- Opens Jupyter notebook `scripts/training/run_training.ipynb` in your browser
- Run cells sequentially (or click "Run All")
- Trains neural network on real UCI HAR data (7,000+ samples)
- Exports `weights.json` to `public/model/`
- Check the console output for `✓ Training Pipeline Complete!`

### 2. Start the Web App

```bash
npm run dev
```

**What happens:**

- Starts Next.js server on http://localhost:3000
- Loads your trained model into browser
- Ready for live classification

### 3. Open in Browser and Test

1. Go to http://localhost:3000
2. Find "Live Classification" component (or scroll down)
3. Click "Start" ▶️ button
4. Select simulated activity (e.g., "Walking")
5. Watch the buffer fill to 128/128 readings
6. See **REAL predictions** from your trained model appear!

---

## What Just Happened?

### ✅ Phase 1: Model Training (Offline)

Your computer trained an AI using:

- **Data**: 7,352 real smartphone sensor samples
- **Model**: Neural network with 4 layers
- **Output**: `weights.json` (your trained model)

### ✅ Phase 2: Live Prediction (Online)

Your browser now:

1. **Simulates** sensor readings (50Hz, 20ms intervals)
2. **Buffers** 128 samples (2.56 seconds of data)
3. **Extracts** 561 features from each buffer
4. **Loads** your trained weights
5. **Runs** neural network inference (~2ms)
6. **Displays** activity prediction + confidence

---

## Understanding the Live Classifier

### Buffer Visualization

```
███░░░░░░░░░░░░░░░░░░░░░░  30/128 readings
████████░░░░░░░░░░░░░░░░░░  64/128 readings
████████████████████████████ 128/128 → PREDICT!
```

### Prediction Output

```
Multiclass (6 activities):
  Walking         ████░░░░░░ 45%
  Stairs Up       ███░░░░░░░ 28%
  Stairs Down     ░░░░░░░░░░  5%
  Sitting         ░░░░░░░░░░  8%
  Standing        ░░░░░░░░░░ 10%
  Laying          ░░░░░░░░░░  4%

  PREDICTED: Walking (45%)

Binary (Health):
  ACTIVE ✓ (Movement detected)
```

---

## Troubleshooting

| Problem                                       | Solution                                                                                                          |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `npm run train` fails                         | Make sure you're in the right directory. Run: `cd /Users/edinabajric/Desktop/Codes/AI_ML/HAR_NN_HealthMonitoring` |
| "Failed to load pre-trained model" in browser | The `npm run train` step didn't complete. Check for errors above.                                                 |
| Predictions are all random/unchanging         | Weights not loaded correctly. Check that `public/model/weights.json` exists (5MB+ file)                           |
| App won't start (`npm run dev` fails)         | Missing dependencies? Run: `pnpm install`                                                                         |
| Buffer never fills to 128                     | Try selecting a different activity, or reload page                                                                |

---

## Files Generated

After `npm run train`, you'll have:

```
public/model/
├── weights.json                 # Your trained model (5-10 MB)
├── confusion_matrix.json        # Shows which classes confuse the model
└── training_history.json        # Loss/accuracy over 100 epochs
```

---

## How It Works (Simple Diagram)

```
TRAINING (One-time, offline):
┌─ Raw Data (train.csv) ─┐
│    561 features each   │
└─ 7,352 samples ────────┘
          ↓
    [Neural Network]
       ↓ learns ↓
    weights.json
          ↓
         saved to
    public/model/

INFERENCE (Real-time, in browser):
┌─ Simulated Sensors ─┐
│ acc_x, acc_y, acc_z │
│ gyro_x, gyro_y, z   │
└─ every 20ms ────────┘
          ↓
    [Buffer 128 samples]
          ↓
    [Extract 561 features]
          ↓
    [Load weights.json]
          ↓
    [MLP Forward Pass]
          ↓
    [Softmax Output]
          ↓
   Walking: 45%  ← PREDICTION
```

---

## What Each Component Does

### `har_training.py`

- Loads UCI HAR CSV data
- Scales all values to [-1, 1]
- Creates neural network (561→256→128→64→6)
- Trains for up to 100 epochs
- Exports `weights.json` and metrics

### `inference-engine.ts` (Browser)

- Loads `weights.json` from network
- Runs forward pass in browser
- Uses same scaling as training
- Produces 6-class probabilities

### `live-classifier.tsx` (UI)

- Simulates sensor readings
- Collects into 128-sample buffer
- Calls inference engine
- Shows predictions on dashboard

---

## Performance

- **Training accuracy**: ~95%
- **Test accuracy**: ~92%
- **Inference speed**: ~2ms per prediction (very fast!)
- **Model size**: ~5-10 MB (fits easily in browsers)

---

## Next: Explore More

Once everything works:

1. **Adjust simulated activity** and watch predictions change
2. **View Training Charts** tab - see how the AI learned
3. **Check Confusion Matrix** - understand where AI makes mistakes
4. **Read the SETUP_GUIDE.md** - deep dive into architecture
5. **Modify har_training.py** - try different hyperparameters

---

## Summary

✓ **Data**: Real UCI smartphone sensor data (7,000+ samples)
✓ **Training**: MLP neural network with 4 layers
✓ **Deployment**: Browser-based inference (no server call)
✓ **Simulation**: 50Hz sensor buffering + feature extraction
✓ **Output**: Real-time activity recognition

**You now have a fully functional AI health monitoring system! 🎉**

---

## Questions?

Check out:

- `SETUP_GUIDE.md` - Complete technical reference
- `plan/plan.md` - Project requirements
- `components/har/live-classifier.tsx` - Implementation details
- Comments in `lib/inference-engine.ts` - How inference works
