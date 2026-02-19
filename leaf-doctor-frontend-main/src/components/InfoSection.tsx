import { motion } from "framer-motion";
import { Microscope, Database, CloudLightning, ShieldCheck } from "lucide-react";

const infoItems = [
  {
    icon: Microscope,
    title: "Deep Learning Model",
    description:
      "Powered by a convolutional neural network trained on thousands of labeled plant leaf samples across multiple species and disease types.",
  },
  {
    icon: Database,
    title: "Extensive Dataset",
    description:
      "Our model is trained on a curated dataset covering 50+ diseases including blight, rust, mildew, leaf spot, and nutrient deficiencies.",
  },
  {
    icon: CloudLightning,
    title: "Real-Time Inference",
    description:
      "Get predictions in under 3 seconds. The lightweight backend processes images on-the-fly with optimized model serving.",
  },
  {
    icon: ShieldCheck,
    title: "Reliable Results",
    description:
      "Each prediction includes a confidence score so you know how reliable the diagnosis is. High accuracy across diverse conditions.",
  },
];

export function InfoSection() {
  return (
    <section className="space-y-5">
      <div className="text-center space-y-2">
        <p className="text-[10px] uppercase tracking-[0.2em] font-bold text-primary">
          Under the Hood
        </p>
        <h2 className="text-xl font-serif font-bold text-foreground">
          Technology & Approach
        </h2>
        <div className="section-divider mt-2" />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {infoItems.map((item, i) => (
          <motion.div
            key={item.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ delay: i * 0.1, duration: 0.5 }}
            className="glass-card rounded-xl p-4 space-y-3 group hover:border-primary/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className="rounded-lg gradient-hero p-2.5 shadow-sm group-hover:shadow-glow transition-shadow">
                <item.icon className="h-4 w-4 text-primary-foreground" />
              </div>
              <h3 className="text-sm font-bold text-foreground">{item.title}</h3>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {item.description}
            </p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
