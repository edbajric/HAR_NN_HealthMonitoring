'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw, AlertTriangle, Info } from 'lucide-react'
import { 
  Activity, 
  ACTIVITIES, 
  ACTIVITY_LABELS,
  generateSensorSample,
  SensorSample,
  DATASET_STATS
} from '@/lib/har-data'
import { 
  getInferenceEngine, 
  extractFeatures,
  MLPInferenceEngine,
  InferenceResult 
} from '@/lib/inference-engine'

// Constants from UCI HAR specification
const WINDOW_SIZE = 128 // 2.56 seconds at 50Hz
const SAMPLING_RATE_MS = 20 // 50Hz = 20ms per sample
const OVERLAP_RATIO = 0.5 // 50% overlap

interface SensorBuffer {
  acc: { x: number; y: number; z: number }[]
  gyro: { x: number; y: number; z: number }[]
}

export function LiveClassifier() {
  const [isRunning, setIsRunning] = useState(false)
  const [currentActivity, setCurrentActivity] = useState<Activity>('WALKING')
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null)
  const [sensorBuffer, setSensorBuffer] = useState<SensorBuffer>({ acc: [], gyro: [] })
  const [bufferFillPercent, setBufferFillPercent] = useState(0)
  const [engine, setEngine] = useState<MLPInferenceEngine | null>(null)
  const [lastSample, setLastSample] = useState<SensorSample | null>(null)
  const [classificationCount, setClassificationCount] = useState(0)
  const [showValueAlignmentInfo, setShowValueAlignmentInfo] = useState(false)
  
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  
  // Initialize inference engine
  useEffect(() => {
    const initEngine = async () => {
      const eng = getInferenceEngine()
      await eng.initialize()
      setEngine(eng)
    }
    initEngine()
  }, [])
  
  // Collect sensor readings and run inference when buffer is full
  const collectSensorReading = useCallback(() => {
    // Generate sensor sample for the simulated activity
    const sample = generateSensorSample(currentActivity)
    setLastSample(sample)
    
    setSensorBuffer(prev => {
      const newAcc = [...prev.acc, { x: sample.accX, y: sample.accY, z: sample.accZ }]
      const newGyro = [...prev.gyro, { x: sample.gyroX, y: sample.gyroY, z: sample.gyroZ }]
      
      // Keep only the last WINDOW_SIZE readings (sliding window)
      const trimmedAcc = newAcc.slice(-WINDOW_SIZE)
      const trimmedGyro = newGyro.slice(-WINDOW_SIZE)
      
      // Update buffer fill percentage
      const fillPercent = Math.min(100, (trimmedAcc.length / WINDOW_SIZE) * 100)
      setBufferFillPercent(fillPercent)
      
      // Run inference when buffer is full
      if (trimmedAcc.length >= WINDOW_SIZE && engine) {
        try {
          const features = extractFeatures(trimmedAcc, trimmedGyro)
          const result = engine.predict(features)
          setInferenceResult(result)
          setClassificationCount(c => c + 1)
        } catch (err) {
          console.error('[v0] Inference error:', err)
        }
      }
      
      return { acc: trimmedAcc, gyro: trimmedGyro }
    })
  }, [currentActivity, engine])
  
  // Start/stop sensor collection
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(collectSensorReading, SAMPLING_RATE_MS)
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isRunning, collectSensorReading])
  
  const resetBuffer = () => {
    setSensorBuffer({ acc: [], gyro: [] })
    setBufferFillPercent(0)
    setInferenceResult(null)
    setClassificationCount(0)
  }
  
  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg text-foreground">Live Classification</CardTitle>
            <CardDescription>MLP Inference with 128-Sample Window</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="icon"
              onClick={() => setIsRunning(!isRunning)}
              title={isRunning ? 'Pause' : 'Start'}
            >
              {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
            <Button 
              variant="ghost" 
              size="icon"
              onClick={resetBuffer}
              title="Reset buffer"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Buffer Status */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">
              Window Buffer ({sensorBuffer.acc.length}/{WINDOW_SIZE})
            </p>
            <span className="text-xs font-mono text-muted-foreground">
              {DATASET_STATS.windowSize}s @ {DATASET_STATS.samplingRate}Hz
            </span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-100 ${
                bufferFillPercent >= 100 ? 'bg-success' : 'bg-primary'
              }`}
              style={{ width: `${bufferFillPercent}%` }}
            />
          </div>
          {bufferFillPercent < 100 && (
            <p className="text-xs text-muted-foreground mt-1">
              Collecting {WINDOW_SIZE - sensorBuffer.acc.length} more readings...
            </p>
          )}
        </div>
        
        {/* Activity Selector */}
        <div>
          <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Simulated Activity</p>
          <div className="flex flex-wrap gap-1.5">
            {ACTIVITIES.map((activity) => (
              <button
                key={activity}
                onClick={() => setCurrentActivity(activity)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  currentActivity === activity
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                {ACTIVITY_LABELS[activity]}
              </button>
            ))}
          </div>
        </div>
        
        {/* Softmax Probability Distribution */}
        {inferenceResult && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">
                Softmax Probabilities
              </p>
              <span className="text-xs font-mono text-muted-foreground">
                {inferenceResult.inferenceTimeMs.toFixed(2)}ms
              </span>
            </div>
            <div className="space-y-1.5">
              {ACTIVITIES.map((activity) => {
                const prob = inferenceResult.multiclass.probabilities[activity]
                const isTop = activity === inferenceResult.multiclass.predicted
                return (
                  <div key={activity} className="flex items-center gap-2">
                    <span className={`text-xs w-20 truncate ${
                      isTop ? 'font-semibold text-foreground' : 'text-muted-foreground'
                    }`}>
                      {ACTIVITY_LABELS[activity]}
                    </span>
                    <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-200 ${
                          isTop ? 'bg-primary' : 'bg-muted-foreground/30'
                        }`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className={`text-xs font-mono w-12 text-right ${
                      isTop ? 'text-foreground' : 'text-muted-foreground'
                    }`}>
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}
        
        {/* Prediction Display */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-muted/50 border border-border">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Multiclass</p>
            <p className="text-lg font-semibold text-foreground">
              {inferenceResult ? ACTIVITY_LABELS[inferenceResult.multiclass.predicted] : '—'}
            </p>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${(inferenceResult?.multiclass.confidence ?? 0) * 100}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground font-mono">
                {inferenceResult ? (inferenceResult.multiclass.confidence * 100).toFixed(0) : 0}%
              </span>
            </div>
          </div>
          
          <div className={`p-3 rounded-lg border ${
            inferenceResult?.binary.predicted === 'ACTIVE' 
              ? 'bg-success/10 border-success/20' 
              : 'bg-warning/10 border-warning/20'
          }`}>
            <div className="flex items-center justify-between">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Binary</p>
              <button
                onClick={() => setShowValueAlignmentInfo(!showValueAlignmentInfo)}
                className="text-muted-foreground hover:text-foreground"
              >
                <Info className="w-3 h-3" />
              </button>
            </div>
            <p className={`text-lg font-semibold ${
              inferenceResult?.binary.predicted === 'ACTIVE' ? 'text-success' : 'text-warning'
            }`}>
              {inferenceResult?.binary.predicted ?? '—'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {inferenceResult?.binary.predicted === 'ACTIVE' ? 'Movement detected' : 'Stationary state'}
            </p>
          </div>
        </div>
        
        {/* Value Alignment Info */}
        {showValueAlignmentInfo && (
          <div className="p-3 rounded-lg bg-warning/5 border border-warning/20">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-warning mt-0.5 shrink-0" />
              <div className="text-xs text-muted-foreground">
                <p className="font-medium text-foreground mb-1">Value Alignment: Sedentary Bias</p>
                <p>
                  To minimize false negatives for sedentary behavior (a health risk), 
                  the agent prioritizes SEDENTARY detection when combined sedentary 
                  probability exceeds 25%. This addresses the asymmetric cost of 
                  misclassification in health monitoring.
                </p>
                {inferenceResult && (
                  <div className="mt-2 flex gap-4 font-mono">
                    <span>Active: {(inferenceResult.binary.activeProb * 100).toFixed(1)}%</span>
                    <span>Sedentary: {(inferenceResult.binary.sedentaryProb * 100).toFixed(1)}%</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        
        {/* Sensor Stream */}
        <div>
          <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">
            Sensor Stream ({DATASET_STATS.samplingRate}Hz)
          </p>
          <div className="grid grid-cols-3 gap-2">
            <SensorGauge label="Acc X" value={lastSample?.accX ?? 0} unit="g" />
            <SensorGauge label="Acc Y" value={lastSample?.accY ?? 0} unit="g" />
            <SensorGauge label="Acc Z" value={lastSample?.accZ ?? 0} unit="g" />
          </div>
          <div className="grid grid-cols-3 gap-2 mt-2">
            <SensorGauge label="Gyro X" value={lastSample?.gyroX ?? 0} unit="rad/s" />
            <SensorGauge label="Gyro Y" value={lastSample?.gyroY ?? 0} unit="rad/s" />
            <SensorGauge label="Gyro Z" value={lastSample?.gyroZ ?? 0} unit="rad/s" />
          </div>
        </div>
        
        {/* Classification Stats */}
        <div className="flex items-center justify-between text-xs text-muted-foreground border-t border-border pt-3">
          <span>Classifications: {classificationCount}</span>
          <span>Window Overlap: {DATASET_STATS.overlap}%</span>
        </div>
      </CardContent>
    </Card>
  )
}

function SensorGauge({ label, value, unit }: { label: string; value: number; unit: string }) {
  const normalizedValue = Math.min(1, Math.max(-1, value))
  const percentage = ((normalizedValue + 1) / 2) * 100
  
  return (
    <div className="p-2 rounded-lg bg-muted/30">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-muted-foreground font-mono">{label}</span>
        <span className="text-[10px] text-muted-foreground">{unit}</span>
      </div>
      <div className="text-xs text-foreground font-mono mb-1">{value.toFixed(3)}</div>
      <div className="h-1 bg-muted rounded-full overflow-hidden relative">
        {/* Center line */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
        <div 
          className="h-full bg-chart-1 transition-all duration-100 absolute"
          style={{ 
            left: percentage < 50 ? `${percentage}%` : '50%',
            width: Math.abs(percentage - 50) + '%'
          }}
        />
      </div>
    </div>
  )
}
