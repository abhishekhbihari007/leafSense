import { motion } from "framer-motion";
import { Target, Clock, Bug, Users } from "lucide-react";

const stats = [
  { icon: Target, value: "95%+", label: "Accuracy", color: "text-success" },
  { icon: Clock, value: "<3s", label: "Response", color: "text-primary" },
  { icon: Bug, value: "50+", label: "Diseases", color: "text-accent" },
  { icon: Users, value: "10K+", label: "Scans Done", color: "text-primary" },
];

export function StatsBar() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4, duration: 0.6 }}
      className="grid grid-cols-4 gap-3"
    >
      {stats.map((stat, i) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 + i * 0.1 }}
          className="glass-card-solid rounded-xl p-3 text-center space-y-1"
        >
          <stat.icon className={`h-5 w-5 mx-auto ${stat.color}`} />
          <p className={`text-xl font-heading font-normal ${stat.color}`}>{stat.value}</p>
          <p className="text-xs font-subtitle uppercase tracking-[0.15em] text-muted-foreground font-semibold">
            {stat.label}
          </p>
        </motion.div>
      ))}
    </motion.div>
  );
}
