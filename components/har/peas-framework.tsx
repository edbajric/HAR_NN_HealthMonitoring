'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Target, Globe, Zap, Radio } from 'lucide-react'

const peasComponents = [
  {
    letter: 'P',
    title: 'Performance',
    description: 'High classification accuracy with minimal false negatives for sedentary behavior detection',
    icon: Target,
    metrics: ['Accuracy > 95%', 'Recall > 97%', 'F1 Score > 96%']
  },
  {
    letter: 'E',
    title: 'Environment',
    description: 'Physical movements in 3D world - partially observable, stochastic, continuous',
    icon: Globe,
    metrics: ['3D Motion Space', 'Stochastic', 'Continuous']
  },
  {
    letter: 'A',
    title: 'Actuators',
    description: 'Web-based dashboard outputting predicted activity labels and performance metrics',
    icon: Zap,
    metrics: ['Real-time Display', 'Visual Analytics', 'API Output']
  },
  {
    letter: 'S',
    title: 'Sensors',
    description: 'Smartphone accelerometer and gyroscope providing 3-axis signal values at 50Hz',
    icon: Radio,
    metrics: ['Accelerometer', 'Gyroscope', '50Hz Sample Rate']
  }
]

export function PEASFramework() {
  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-foreground">Task Environment (PEAS)</CardTitle>
        <CardDescription>Agent specification following the Standard Model of AI</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {peasComponents.map((component) => (
            <div 
              key={component.letter}
              className="p-4 rounded-xl bg-muted/30 border border-border hover:border-primary/30 transition-colors"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
                  <span className="text-lg font-bold text-primary">{component.letter}</span>
                </div>
                <div>
                  <h3 className="font-semibold text-foreground text-sm">{component.title}</h3>
                </div>
              </div>
              
              <p className="text-xs text-muted-foreground mb-3 leading-relaxed">
                {component.description}
              </p>
              
              <div className="flex flex-wrap gap-1.5">
                {component.metrics.map((metric, i) => (
                  <span 
                    key={i}
                    className="px-2 py-0.5 rounded-full bg-muted text-[10px] text-muted-foreground"
                  >
                    {metric}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
