'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { ArrowRight, Cpu, Database, Brain, Monitor } from 'lucide-react'

export function ArchitectureDiagram() {
  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-foreground">System Architecture</CardTitle>
        <CardDescription>Hybrid Pipeline: Offline Training + Online Inference</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col lg:flex-row items-center justify-between gap-4 py-4">
          {/* Sensors */}
          <ArchitectureNode
            icon={<Cpu className="w-5 h-5" />}
            title="Sensors"
            subtitle="Accelerometer + Gyroscope"
            color="chart-1"
          />
          
          <ArrowRight className="w-5 h-5 text-muted-foreground rotate-90 lg:rotate-0 shrink-0" />
          
          {/* Feature Extraction */}
          <ArchitectureNode
            icon={<Database className="w-5 h-5" />}
            title="Features"
            subtitle="561-dim Vector"
            color="chart-2"
          />
          
          <ArrowRight className="w-5 h-5 text-muted-foreground rotate-90 lg:rotate-0 shrink-0" />
          
          {/* Neural Network */}
          <ArchitectureNode
            icon={<Brain className="w-5 h-5" />}
            title="MLP"
            subtitle="ReLU + Softmax"
            color="chart-3"
          />
          
          <ArrowRight className="w-5 h-5 text-muted-foreground rotate-90 lg:rotate-0 shrink-0" />
          
          {/* Output */}
          <ArchitectureNode
            icon={<Monitor className="w-5 h-5" />}
            title="Output"
            subtitle="Activity Label"
            color="chart-4"
          />
        </div>
        
        {/* Technical Details */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 pt-4 border-t border-border">
          <TechDetail label="Window Size" value="2.56s" />
          <TechDetail label="Overlap" value="50%" />
          <TechDetail label="Hidden Layers" value="3" />
          <TechDetail label="Optimizer" value="SGD" />
        </div>
      </CardContent>
    </Card>
  )
}

function ArchitectureNode({ 
  icon, 
  title, 
  subtitle, 
  color 
}: { 
  icon: React.ReactNode
  title: string
  subtitle: string
  color: string
}) {
  return (
    <div className={`flex flex-col items-center p-4 rounded-xl bg-${color}/10 border border-${color}/20 min-w-[100px]`}
      style={{
        backgroundColor: `oklch(from var(--${color}) l c h / 0.1)`,
        borderColor: `oklch(from var(--${color}) l c h / 0.2)`
      }}
    >
      <div 
        className="p-2 rounded-lg mb-2"
        style={{ backgroundColor: `oklch(from var(--${color}) l c h / 0.2)` }}
      >
        <div style={{ color: `var(--${color})` }}>{icon}</div>
      </div>
      <p className="text-sm font-semibold text-foreground">{title}</p>
      <p className="text-[10px] text-muted-foreground text-center">{subtitle}</p>
    </div>
  )
}

function TechDetail({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center p-2 rounded-lg bg-muted/30">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</p>
      <p className="text-sm font-mono font-semibold text-foreground">{value}</p>
    </div>
  )
}
