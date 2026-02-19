import { motion } from "framer-motion";
import { Upload, Cpu, BarChart3, ChevronRight } from "lucide-react";

const steps = [
  {
    icon: Upload,
    number: "01",
    title: "Upload Image",
    description: "Take or upload a clear photo of your plant's leaf or affected area",
  },
  {
    icon: Cpu,
    number: "02",
    title: "AI Analysis",
    description: "Our deep learning model processes and analyzes the image instantly",
  },
  {
    icon: BarChart3,
    number: "03",
    title: "Get Results",
    description: "Receive a detailed diagnosis with confidence scores and health metrics",
  },
];

export function HowItWorks() {
  return (
    <section className="space-y-6">
      <div className="text-center space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] font-semibold font-subtitle text-primary">
          Simple Process
        </p>
        <h2 className="text-2xl sm:text-3xl font-heading font-normal text-foreground">
          How It Works
        </h2>
        <div className="section-divider mt-1" />
      </div>

      <div className="relative">
        {/* Connecting line */}
        <div className="absolute top-10 left-[15%] right-[15%] h-[2px] bg-gradient-to-r from-primary/10 via-primary/30 to-primary/10 hidden sm:block" />

        <div className="flex items-start justify-between gap-6 sm:gap-8">
          {steps.map((step, i) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 + i * 0.15, duration: 0.5 }}
              className="flex flex-col items-center text-center flex-1 relative group min-w-0"
            >
              <div className="relative mb-4">
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl gradient-hero flex items-center justify-center shadow-elevated z-10 relative"
                >
                  <step.icon className="h-7 w-7 text-primary-foreground" />
                </motion.div>
                {/* Pulse ring */}
                <div className="absolute inset-0 rounded-2xl gradient-hero opacity-15 animate-ping" style={{ animationDuration: "3s" }} />
                {/* Number badge */}
                <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-accent flex items-center justify-center z-20">
                  <span className="text-[10px] font-bold font-subtitle text-accent-foreground">{step.number}</span>
                </div>
              </div>
              <p className="text-sm font-semibold font-subtitle text-foreground mt-1">{step.title}</p>
              <p className="text-xs font-subtitle text-muted-foreground leading-relaxed mt-2 max-w-[180px] sm:max-w-[200px] mx-auto">
                {step.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
