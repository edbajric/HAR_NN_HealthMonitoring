'use client'

import { TrendingUp, Users, Layers, Clock } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { DATASET_STATS } from '@/lib/har-data'

const stats = [
  {
    label: 'Total Samples',
    value: DATASET_STATS.totalSamples.toLocaleString(),
    icon: Layers,
    color: 'text-chart-1',
    bgColor: 'bg-chart-1/10',
    borderColor: 'border-chart-1/20'
  },
  {
    label: 'Features',
    value: DATASET_STATS.features.toString(),
    icon: TrendingUp,
    color: 'text-chart-2',
    bgColor: 'bg-chart-2/10',
    borderColor: 'border-chart-2/20'
  },
  {
    label: 'Subjects',
    value: DATASET_STATS.subjects.toString(),
    icon: Users,
    color: 'text-chart-3',
    bgColor: 'bg-chart-3/10',
    borderColor: 'border-chart-3/20'
  },
  {
    label: 'Sampling Rate',
    value: `${DATASET_STATS.samplingRate} Hz`,
    icon: Clock,
    color: 'text-chart-4',
    bgColor: 'bg-chart-4/10',
    borderColor: 'border-chart-4/20'
  }
]

export function StatsCards() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">{stat.label}</p>
                <p className="text-2xl font-bold mt-1 text-foreground">{stat.value}</p>
              </div>
              <div className={`p-2 rounded-lg ${stat.bgColor} border ${stat.borderColor}`}>
                <stat.icon className={`w-4 h-4 ${stat.color}`} />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
