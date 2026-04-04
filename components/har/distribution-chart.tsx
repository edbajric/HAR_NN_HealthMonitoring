'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import { CLASS_DISTRIBUTION } from '@/lib/har-data'

export function DistributionChart() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg text-foreground">Class Distribution</CardTitle>
          <CardDescription>Sample counts per activity class</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[250px] w-full flex items-center justify-center text-muted-foreground">
            Loading chart...
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-foreground">Class Distribution</CardTitle>
        <CardDescription>Sample counts per activity class</CardDescription>
      </CardHeader>
      <CardContent>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={CLASS_DISTRIBUTION} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0 0)" />
              <XAxis
                dataKey="activity"
                tick={{ fill: 'oklch(0.6 0 0)', fontSize: 10 }}
                axisLine={{ stroke: 'oklch(0.25 0 0)' }}
                tickLine={{ stroke: 'oklch(0.25 0 0)' }}
              />
              <YAxis
                tick={{ fill: 'oklch(0.6 0 0)', fontSize: 10 }}
                axisLine={{ stroke: 'oklch(0.25 0 0)' }}
                tickLine={{ stroke: 'oklch(0.25 0 0)' }}
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
              <Bar
                dataKey="train"
                fill="oklch(0.7 0.18 250)"
                name="Training Set"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="test"
                fill="oklch(0.65 0.16 160)"
                name="Test Set"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
