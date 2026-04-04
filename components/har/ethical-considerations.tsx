'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { 
  AlertTriangle, 
  Users, 
  ShieldAlert, 
  Scale, 
  Brain,
  ChevronDown,
  ChevronUp
} from 'lucide-react'

interface ConsiderationItem {
  id: string
  title: string
  icon: React.ReactNode
  description: string
  details: string[]
  severity: 'high' | 'medium' | 'low'
}

const CONSIDERATIONS: ConsiderationItem[] = [
  {
    id: 'king-midas',
    title: 'King Midas Problem',
    icon: <AlertTriangle className="w-5 h-5" />,
    description: 'Risk of the agent pursuing its objective in ways harmful to human oversight.',
    details: [
      'The model optimizes for classification accuracy, not health outcomes',
      'Misalignment between metric maximization and clinical utility',
      'Agent may prioritize confident predictions over cautious ones',
      'Must ensure model serves as tool to assist, not replace, clinical judgment'
    ],
    severity: 'high'
  },
  {
    id: 'algorithmic-bias',
    title: 'Algorithmic Bias',
    icon: <Users className="w-5 h-5" />,
    description: 'Dataset demographics may not represent all populations.',
    details: [
      'UCI dataset: 30 volunteers aged 19-48 only',
      'May exhibit degraded performance on older populations',
      'Smartphone placement (waist) may differ in real deployments',
      'No representation of physical disabilities or mobility aids',
      'Gender and body type diversity not documented'
    ],
    severity: 'high'
  },
  {
    id: 'asymmetric-cost',
    title: 'Asymmetric Misclassification',
    icon: <Scale className="w-5 h-5" />,
    description: 'False negatives for sedentary behavior carry higher health risks.',
    details: [
      'Standard cross-entropy treats all errors equally',
      'Missing sedentary detection could overlook health risks',
      'Value alignment requires biasing toward sedentary detection',
      'Implemented 25% threshold bias for SEDENTARY classification',
      'Prioritizes sensitivity over specificity for health safety'
    ],
    severity: 'medium'
  },
  {
    id: 'data-drift',
    title: 'Covariate Shift',
    icon: <ShieldAlert className="w-5 h-5" />,
    description: 'Real-world performance degrades as input statistics change over time.',
    details: [
      'Model trained on controlled lab conditions',
      'Real-world sensor noise patterns may differ',
      'Device hardware variations affect signal characteristics',
      'Seasonal and environmental factors not captured',
      'Requires ongoing monitoring and recalibration'
    ],
    severity: 'medium'
  },
  {
    id: 'automation-bias',
    title: 'Automation Bias',
    icon: <Brain className="w-5 h-5" />,
    description: 'Over-reliance on algorithmic outputs without critical evaluation.',
    details: [
      'Risk of treating model output as "objective truth"',
      'Mathematical precision does not guarantee correctness',
      'Human oversight must not atrophy with automation',
      'Model provides probability estimates, not certainties',
      'Critical thinking required for clinical decisions'
    ],
    severity: 'medium'
  }
]

export function EthicalConsiderations() {
  const [expandedItems, setExpandedItems] = useState<string[]>(['king-midas'])
  
  const toggleItem = (id: string) => {
    setExpandedItems(prev => 
      prev.includes(id) 
        ? prev.filter(i => i !== id)
        : [...prev, id]
    )
  }
  
  const severityColors = {
    high: 'border-destructive/30 bg-destructive/5',
    medium: 'border-warning/30 bg-warning/5',
    low: 'border-muted bg-muted/30'
  }
  
  const severityTextColors = {
    high: 'text-destructive',
    medium: 'text-warning',
    low: 'text-muted-foreground'
  }
  
  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-warning/10 border border-warning/20">
            <ShieldAlert className="w-5 h-5 text-warning" />
          </div>
          <div>
            <CardTitle className="text-lg text-foreground">
              Ethical Considerations & Professional Integrity
            </CardTitle>
            <CardDescription>
              Critical analysis of value alignment and potential risks
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {CONSIDERATIONS.map((item) => {
          const isExpanded = expandedItems.includes(item.id)
          
          return (
            <div 
              key={item.id}
              className={`rounded-lg border transition-colors ${severityColors[item.severity]}`}
            >
              <button
                onClick={() => toggleItem(item.id)}
                className="w-full p-3 flex items-start gap-3 text-left"
              >
                <div className={`mt-0.5 ${severityTextColors[item.severity]}`}>
                  {item.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="font-medium text-foreground">{item.title}</h4>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs uppercase tracking-wider ${severityTextColors[item.severity]}`}>
                        {item.severity} risk
                      </span>
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-muted-foreground" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-muted-foreground" />
                      )}
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    {item.description}
                  </p>
                </div>
              </button>
              
              {isExpanded && (
                <div className="px-3 pb-3 pt-0">
                  <div className="ml-8 pl-3 border-l-2 border-border">
                    <ul className="space-y-1.5">
                      {item.details.map((detail, idx) => (
                        <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                          <span className="text-primary mt-1.5 shrink-0">•</span>
                          <span>{detail}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )
        })}
        
        {/* Transparency Statement */}
        <div className="mt-4 p-3 rounded-lg bg-muted/30 border border-border">
          <p className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Transparency Statement:</span>{' '}
            This rational agent provides probabilistic classifications based on statistical 
            patterns learned from 30 volunteers. It is designed to assist human oversight, 
            not replace clinical judgment. The softmax probability distribution is displayed 
            to communicate model uncertainty and enable informed decision-making.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
