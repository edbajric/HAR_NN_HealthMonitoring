import { Header } from '@/components/har/header'
import { StatsCards } from '@/components/har/stats-cards'
import { PEASFramework } from '@/components/har/peas-framework'
import { ArchitectureDiagram } from '@/components/har/architecture-diagram'
import { LiveClassifier } from '@/components/har/live-classifier'
import { ModelComparison } from '@/components/har/model-comparison'
import { LossChart, AccuracyChart } from '@/components/har/training-charts'
import { DistributionChart } from '@/components/har/distribution-chart'
import { EthicalConsiderations } from '@/components/har/ethical-considerations'

export default function HARDashboard() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container mx-auto px-4 py-6 space-y-6">
        {/* Hero Section */}
        <section className="text-center py-8">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-3 text-balance">
            Human Activity Recognition Agent
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto text-pretty">
            A rational agent designed to classify human physical activity using neural networks 
            on the UCI HAR dataset. Implements both multiclass (6 activities) and binary 
            (Active/Sedentary) classification for health monitoring.
          </p>
        </section>
        
        {/* Dataset Stats */}
        <section>
          <StatsCards />
        </section>
        
        {/* PEAS Framework */}
        <section>
          <PEASFramework />
        </section>
        
        {/* Architecture */}
        <section>
          <ArchitectureDiagram />
        </section>
        
        {/* Live Classifier + Training Charts */}
        <section className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <LiveClassifier />
          </div>
          <div className="lg:col-span-2 grid md:grid-cols-2 gap-6">
            <LossChart />
            <AccuracyChart />
          </div>
        </section>
        
        {/* Model Comparison */}
        <section>
          <ModelComparison />
        </section>
        
        {/* Distribution Chart */}
        <section>
          <DistributionChart />
        </section>
        
        {/* Ethical Considerations */}
        <section>
          <EthicalConsiderations />
        </section>
        
        {/* Footer */}
        <footer className="border-t border-border pt-6 mt-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
            <div>
              <p className="font-medium text-foreground">UCI HAR Dataset</p>
              <p className="text-xs">
                Human Activity Recognition Using Smartphones Dataset
              </p>
            </div>
            <div className="flex items-center gap-6 text-xs">
              <div>
                <span className="text-muted-foreground">Subjects:</span>
                <span className="ml-1 text-foreground font-medium">30</span>
              </div>
              <div>
                <span className="text-muted-foreground">Activities:</span>
                <span className="ml-1 text-foreground font-medium">6</span>
              </div>
              <div>
                <span className="text-muted-foreground">Features:</span>
                <span className="ml-1 text-foreground font-medium">561</span>
              </div>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
