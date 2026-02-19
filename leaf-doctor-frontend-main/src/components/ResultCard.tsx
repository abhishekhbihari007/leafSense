import { motion } from "framer-motion";
import type { PredictionResult } from "@/lib/api";
import { CheckCircle2, AlertTriangle, Leaf, Sparkles, Shield, TrendingUp } from "lucide-react";

interface ResultCardProps {
  result: PredictionResult;
}

export function ResultCard({ result }: ResultCardProps) {
  const isHealthy = result.class?.toLowerCase() === "healthy";

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="space-y-4"
    >
      {/* Main result banner */}
      <div
        className={`relative overflow-hidden rounded-2xl p-6 ${
          isHealthy
            ? "bg-gradient-to-br from-success/15 via-success/5 to-transparent border border-success/25"
            : "bg-gradient-to-br from-destructive/15 via-destructive/5 to-transparent border border-destructive/25"
        }`}
      >
        {/* Background decorative circle */}
        <motion.div
          className={`absolute -right-10 -top-10 h-40 w-40 rounded-full blur-3xl ${
            isHealthy ? "bg-success/10" : "bg-destructive/10"
          }`}
          animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
          transition={{ duration: 4, repeat: Infinity }}
        />

        <div className="relative flex items-center gap-4">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
            className={`flex-shrink-0 rounded-xl p-3.5 ${
              isHealthy
                ? "bg-success/20 text-success shadow-[0_0_24px_hsl(155_60%_38%/0.2)]"
                : "bg-destructive/20 text-destructive shadow-[0_0_24px_hsl(0_72%_51%/0.2)]"
            }`}
          >
            {isHealthy ? (
              <CheckCircle2 className="h-8 w-8" />
            ) : (
              <AlertTriangle className="h-8 w-8" />
            )}
          </motion.div>
          <div>
            <p className="text-xs font-semibold font-subtitle uppercase tracking-[0.15em] text-muted-foreground mb-1">
              Diagnosis Result
            </p>
            <motion.p
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className={`text-3xl sm:text-4xl font-heading font-normal ${
                isHealthy ? "text-success" : "text-destructive"
              }`}
            >
              {result.class}
            </motion.p>
            {result.message && (
              <p className="mt-1 text-sm font-subtitle text-muted-foreground max-w-md">
                {result.message}
              </p>
            )}
          </div>
        </div>

        {/* Status badge */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className={`mt-4 inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-subtitle font-semibold ${
            isHealthy
              ? "bg-success/10 text-success border border-success/20"
              : "bg-destructive/10 text-destructive border border-destructive/20"
          }`}
        >
          <TrendingUp className="h-3 w-3" />
          {isHealthy ? "Your plant looks great!" : "Disease indicators detected"}
        </motion.div>
        {result.recommendation && (
          <p className="mt-3 text-xs font-subtitle text-muted-foreground leading-relaxed border-l-2 border-primary/30 pl-3">
            {result.recommendation}
          </p>
        )}
      </div>

      {/* Confidence meter */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-card rounded-2xl p-5 space-y-3"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold font-subtitle text-foreground">Confidence Level</span>
          </div>
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className={`text-2xl font-heading font-normal ${
              result.confidence > 75 ? "text-success" : result.confidence > 50 ? "text-accent" : "text-destructive"
            }`}
          >
            {result.confidence.toFixed(1)}%
          </motion.span>
        </div>

        {/* Animated progress bar */}
        <div className="relative h-4 w-full overflow-hidden rounded-full bg-muted">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(result.confidence, 100)}%` }}
            transition={{ duration: 1.2, ease: "easeOut", delay: 0.4 }}
            className={`h-full rounded-full ${
              isHealthy
                ? "bg-gradient-to-r from-primary to-success"
                : "bg-gradient-to-r from-destructive to-accent"
            }`}
          />
          {/* Shimmer */}
          <div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-primary-foreground/10 to-transparent animate-shimmer"
            style={{ backgroundSize: "200% 100%" }}
          />
        </div>

        <p className="text-xs font-subtitle text-muted-foreground text-center">
          {result.confidence_tier === "high" || result.confidence > 85
            ? "✓ Very high confidence — reliable result"
            : result.confidence_tier === "moderate" || result.confidence > 60
            ? "Moderate confidence — result is indicative; consider a clearer close-up for best accuracy"
            : "Lower confidence — try a clearer, close-up photo of the leaf for a more accurate result"}
        </p>
      </motion.div>

      {/* Extra scores */}
      {(result.nutrient_score != null || result.random_confidence_score != null) && (
        <div className="grid grid-cols-2 gap-3">
          {result.nutrient_score != null && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              whileHover={{ y: -3 }}
              className="glass-card rounded-xl p-4 space-y-2"
            >
              <div className="flex items-center gap-2">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Leaf className="h-4 w-4 text-primary" />
                </div>
                <span className="text-xs font-semibold font-subtitle text-muted-foreground">Nutrient Score</span>
              </div>
              <p className="text-2xl font-heading font-normal text-foreground">
                {result.nutrient_score}
                <span className="text-xs font-sans text-muted-foreground ml-0.5">/100</span>
              </p>
            </motion.div>
          )}
          {result.random_confidence_score != null && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 }}
              whileHover={{ y: -3 }}
              className="glass-card rounded-xl p-4 space-y-2"
            >
              <div className="flex items-center gap-2">
                <div className="rounded-lg bg-accent/15 p-2">
                  <Sparkles className="h-4 w-4 text-accent" />
                </div>
                <span className="text-xs font-semibold font-subtitle text-muted-foreground">Extra Score</span>
              </div>
              <p className="text-2xl font-heading font-normal text-foreground">
                {result.random_confidence_score}
              </p>
            </motion.div>
          )}
        </div>
      )}
    </motion.div>
  );
}
