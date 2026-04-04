'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ConfusionMatrix } from './confusion-matrix'
import { Check, X, TrendingUp, TrendingDown } from 'lucide-react'

export function ModelComparison() {
  const [selectedModel, setSelectedModel] = useState<'nn' | 'lr'>('nn')
  
  const modelStats = {
    nn: {
      name: 'Neural Network (MLP)',
      accuracy: 95.2,
      params: '45,318',
      trainTime: '2.3s',
      inference: '0.8ms',
      pros: ['High accuracy', 'Non-linear patterns', 'Feature abstraction'],
      cons: ['More parameters', 'Requires tuning']
    },
    lr: {
      name: 'Logistic Regression',
      accuracy: 86.7,
      params: '3,366',
      trainTime: '0.4s',
      inference: '0.2ms',
      pros: ['Fast training', 'Interpretable', 'Few parameters'],
      cons: ['Linear only', 'Lower accuracy']
    }
  }
  
  return (
    <div className="space-y-4">
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg text-foreground">Model Comparison</CardTitle>
          <CardDescription>Neural Network vs Logistic Regression Baseline</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={selectedModel} onValueChange={(v) => setSelectedModel(v as 'nn' | 'lr')}>
            <TabsList className="grid w-full grid-cols-2 mb-4">
              <TabsTrigger value="nn" className="text-xs">
                Neural Network
                <span className="ml-2 px-1.5 py-0.5 rounded bg-success/20 text-success text-[10px]">
                  95.2%
                </span>
              </TabsTrigger>
              <TabsTrigger value="lr" className="text-xs">
                Logistic Regression
                <span className="ml-2 px-1.5 py-0.5 rounded bg-warning/20 text-warning text-[10px]">
                  86.7%
                </span>
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="nn" className="mt-0">
              <ModelDetails stats={modelStats.nn} />
            </TabsContent>
            
            <TabsContent value="lr" className="mt-0">
              <ModelDetails stats={modelStats.lr} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
      
      {/* Side by side confusion matrices */}
      <div className="grid md:grid-cols-2 gap-4">
        <ConfusionMatrix model="nn" />
        <ConfusionMatrix model="lr" />
      </div>
    </div>
  )
}

function ModelDetails({ stats }: { stats: {
  name: string
  accuracy: number
  params: string
  trainTime: string
  inference: string
  pros: string[]
  cons: string[]
}}) {
  const improvement = stats.accuracy > 90
  
  return (
    <div className="space-y-4">
      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard 
          label="Accuracy" 
          value={`${stats.accuracy}%`} 
          trend={improvement ? 'up' : 'down'} 
        />
        <StatCard label="Parameters" value={stats.params} />
        <StatCard label="Train Time" value={stats.trainTime} />
        <StatCard label="Inference" value={stats.inference} />
      </div>
      
      {/* Pros and Cons */}
      <div className="grid md:grid-cols-2 gap-3">
        <div className="p-3 rounded-lg bg-success/5 border border-success/10">
          <p className="text-xs text-success font-medium mb-2 uppercase tracking-wider">Advantages</p>
          <ul className="space-y-1">
            {stats.pros.map((pro, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-foreground">
                <Check className="w-3 h-3 text-success shrink-0" />
                {pro}
              </li>
            ))}
          </ul>
        </div>
        
        <div className="p-3 rounded-lg bg-destructive/5 border border-destructive/10">
          <p className="text-xs text-destructive font-medium mb-2 uppercase tracking-wider">Limitations</p>
          <ul className="space-y-1">
            {stats.cons.map((con, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-foreground">
                <X className="w-3 h-3 text-destructive shrink-0" />
                {con}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, trend }: { label: string; value: string; trend?: 'up' | 'down' }) {
  return (
    <div className="p-3 rounded-lg bg-muted/30 border border-border">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</p>
      <div className="flex items-center gap-1.5 mt-1">
        <p className="text-lg font-semibold font-mono text-foreground">{value}</p>
        {trend && (
          trend === 'up' 
            ? <TrendingUp className="w-4 h-4 text-success" />
            : <TrendingDown className="w-4 h-4 text-destructive" />
        )}
      </div>
    </div>
  )
}
