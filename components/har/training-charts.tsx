'use client'

import { useMemo, useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import { generateLossData, generateAccuracyData } from '@/lib/har-data'

export function LossChart() {
  const [mounted, setMounted] = useState(false)
  const data = useMemo(() => generateLossData(50), [])

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg text-foreground">Loss Landscape</CardTitle>
          <CardDescription>Cross-Entropy Loss Over Training</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-62.5 w-full flex items-center justify-center text-muted-foreground">
            Loading chart...
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-foreground">Loss Landscape</CardTitle>
        <CardDescription>Cross-Entropy Loss Over Training</CardDescription>
      </CardHeader>
      <CardContent>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0 0)" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: 'oklch(0.6 0 0)', fontSize: 10 }}
                axisLine={{ stroke: 'oklch(0.25 0 0)' }}
                tickLine={{ stroke: 'oklch(0.25 0 0)' }}
              />
              <YAxis
                tick={{ fill: 'oklch(0.6 0 0)', fontSize: 10 }}
                axisLine={{ stroke: 'oklch(0.25 0 0)' }}
                tickLine={{ stroke: 'oklch(0.25 0 0)' }}
                domain={[0, 3]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'oklch(0.14 0 0)',
                  border: '1px solid oklch(0.25 0 0)',
                  borderRadius: '8px',
                  color: 'oklch(0.95 0 0)'
                }}
                labelStyle={{ color: 'oklch(0.6 0 0)' }}
              />
              <Legend
                wrapperStyle={{ fontSize: '12px' }}
              />
              <Line
                type="monotone"
                dataKey="trainLoss"
                stroke="oklch(0.7 0.18 250)"
                strokeWidth={2}
                dot={false}
                name="Training Loss"
              />
              <Line
                type="monotone"
                dataKey="valLoss"
                stroke="oklch(0.75 0.15 50)"
                strokeWidth={2}
                dot={false}
                name="Validation Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

export function AccuracyChart() {
  const [mounted, setMounted] = useState(false)
  const data = useMemo(() => generateAccuracyData(50), [])

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg text-foreground">Accuracy Curve</CardTitle>
          <CardDescription>Classification Accuracy Over Training</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-62.5 w-full flex items-center justify-center text-muted-foreground">
            Loading chart...
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-foreground">Accuracy Curve</CardTitle>
        <CardDescription>Classification Accuracy Over Training</CardDescription>
      </CardHeader>
      <CardContent>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0 0)" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: 'oklch(0.6 0 0)', fontSize: 10 }}
                axisLine={{ stroke: 'oklch(0.25 0 0)' }}
                tickLine={{ stroke: 'oklch(0.25 0 0)' }}
              />
              <YAxis
                tick={{ fill: 'oklch(0.6 0 0)', fontSize: 10 }}
                axisLine={{ stroke: 'oklch(0.25 0 0)' }}
                tickLine={{ stroke: 'oklch(0.25 0 0)' }}
                domain={[0, 1]}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'oklch(0.14 0 0)',
                  border: '1px solid oklch(0.25 0 0)',
                  borderRadius: '8px',
                  color: 'oklch(0.95 0 0)'
                }}
                labelStyle={{ color: 'oklch(0.6 0 0)' }}
                formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
              />
              <Legend
                wrapperStyle={{ fontSize: '12px' }}
              />
              <Line
                type="monotone"
                dataKey="trainAcc"
                stroke="oklch(0.65 0.16 160)"
                strokeWidth={2}
                dot={false}
                name="Training Accuracy"
              />
              <Line
                type="monotone"
                dataKey="valAcc"
                stroke="oklch(0.7 0.2 280)"
                strokeWidth={2}
                dot={false}
                name="Validation Accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
