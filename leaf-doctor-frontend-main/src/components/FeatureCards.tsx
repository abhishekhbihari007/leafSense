import { motion } from "framer-motion";
import { Zap, ShieldCheck, Leaf } from "lucide-react";

const features = [
  {
    icon: Zap,
    title: "Instant Analysis",
    description: "Upload a leaf photo and get results in seconds — no waiting, no complexity",
    iconBg: "bg-accent/15",
    iconColor: "text-accent",
  },
  {
    icon: ShieldCheck,
    title: "High Accuracy",
    description: "Deep learning model trained on thousands of verified samples for reliable diagnosis",
    iconBg: "bg-primary/15",
    iconColor: "text-primary",
  },
  {
    icon: Leaf,
    title: "Multi-Disease",
    description: "Covers 50+ plant diseases across multiple crop species and growing conditions",
    iconBg: "bg-success/15",
    iconColor: "text-success",
  },
];

export function FeatureCards() {
  return (
    <div className="grid grid-cols-3 gap-3">
      {features.map((f, i) => (
        <motion.div
          key={f.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 + i * 0.1, duration: 0.5 }}
          whileHover={{ y: -6, transition: { duration: 0.2 } }}
          className="glass-card rounded-xl p-4 text-center space-y-3 cursor-default group"
        >
          <div className={`mx-auto w-fit rounded-xl ${f.iconBg} p-3 group-hover:shadow-sm transition-shadow`}>
            <f.icon className={`h-5 w-5 ${f.iconColor}`} />
          </div>
          <p className="text-sm font-semibold font-subtitle text-foreground leading-tight">{f.title}</p>
          <p className="text-xs font-subtitle text-muted-foreground leading-relaxed">{f.description}</p>
        </motion.div>
      ))}
    </div>
  );
}
