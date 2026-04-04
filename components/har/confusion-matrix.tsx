'use client';

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import {
  ACTIVITIES,
  ACTIVITY_LABELS,
  ConfusionMatrixData,
  generateNNConfusionMatrix,
  generateLRConfusionMatrix,
  calculateMetrics,
} from '@/lib/har-data';

interface ConfusionMatrixProps {
  model: 'nn' | 'lr';
}

export function ConfusionMatrix({ model }: ConfusionMatrixProps) {
  const confusionData = useMemo(() => {
    return model === 'nn' ? generateNNConfusionMatrix() : generateLRConfusionMatrix();
  }, [model]);

  const metrics = useMemo(() => calculateMetrics(confusionData), [confusionData]);

  // Find max value for color scaling
  const maxCount = Math.max(...confusionData.map(d => d.count));

  const getColorIntensity = (count: number, isCorrect: boolean) => {
    const intensity = count / maxCount;
    if (isCorrect) {
      return `rgba(var(--chart-2-rgb, 100, 200, 150), ${0.2 + intensity * 0.6})`;
    }
    if (count === 0) return 'transparent';
    return `rgba(var(--chart-5-rgb, 200, 100, 100), ${0.1 + intensity * 0.4})`;
  };

  const getCellValue = (actual: string, predicted: string): number => {
    const cell = confusionData.find(d => d.actual === actual && d.predicted === predicted);
    return cell?.count ?? 0;
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-foreground">
          {model === 'nn' ? 'Neural Network' : 'Logistic Regression'}
        </CardTitle>
        <CardDescription>Confusion Matrix - 6-Class Classification</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="min-w-100">
            {/* Header row */}
            <div className="flex mb-1">
              <div className="w-16 shrink-0" />
              {ACTIVITIES.map(activity => (
                <div
                  key={activity}
                  className="flex-1 text-center text-[10px] text-muted-foreground font-medium px-0.5 truncate"
                  title={ACTIVITY_LABELS[activity]}
                >
                  {ACTIVITY_LABELS[activity].slice(0, 4)}
                </div>
              ))}
            </div>

            {/* Matrix rows */}
            {ACTIVITIES.map(actualActivity => (
              <div key={actualActivity} className="flex mb-1">
                <div className="w-16 shrink-0 text-[10px] text-muted-foreground font-medium flex items-center truncate pr-1">
                  {ACTIVITY_LABELS[actualActivity].slice(0, 6)}
                </div>
                {ACTIVITIES.map(predictedActivity => {
                  const count = getCellValue(actualActivity, predictedActivity);
                  const isCorrect = actualActivity === predictedActivity;

                  return (
                    <div
                      key={`${actualActivity}-${predictedActivity}`}
                      className="flex-1 aspect-square flex items-center justify-center text-xs font-mono mx-0.5 rounded"
                      style={{
                        backgroundColor: isCorrect
                          ? `oklch(0.65 0.16 160 / ${0.15 + (count / maxCount) * 0.6})`
                          : count > 0
                            ? `oklch(0.6 0.2 20 / ${0.1 + (count / maxCount) * 0.4})`
                            : 'transparent',
                        color: count > 0 ? 'var(--foreground)' : 'var(--muted-foreground)',
                      }}
                    >
                      {count > 0 ? count : '-'}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-2 mt-4 pt-4 border-t border-border">
          <MetricBadge label="Accuracy" value={metrics.accuracy} />
          <MetricBadge label="Precision" value={metrics.precision} />
          <MetricBadge label="Recall" value={metrics.recall} />
          <MetricBadge label="F1 Score" value={metrics.f1Score} />
        </div>
      </CardContent>
    </Card>
  );
}

function MetricBadge({ label, value }: { label: string; value: number }) {
  const percentage = (value * 100).toFixed(1);
  const isHigh = value > 0.9;

  return (
    <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span
        className={`text-sm font-mono font-semibold ${isHigh ? 'text-success' : 'text-warning'}`}
      >
        {percentage}%
      </span>
    </div>
  );
}
