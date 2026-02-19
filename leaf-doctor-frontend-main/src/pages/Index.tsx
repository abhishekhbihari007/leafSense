import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Leaf, RotateCcw, ScanLine, ArrowRight, Sprout, HeartPulse } from "lucide-react";
import { ImageUpload } from "@/components/ImageUpload";
import { ResultCard } from "@/components/ResultCard";
import { FeatureCards } from "@/components/FeatureCards";
import { HowItWorks } from "@/components/HowItWorks";
import { StatsBar } from "@/components/StatsBar";
import { FloatingLeaves } from "@/components/FloatingLeaves";
import { AnalyzingOverlay } from "@/components/AnalyzingOverlay";
import { predictImage, type PredictionResult } from "@/lib/api";

const Index = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileSelect = useCallback((selectedFile: File) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const prediction = await predictImage(file);
      setResult(prediction);
    } catch (err: any) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [file]);

  return (
    <div className="min-h-screen bg-background relative">
      <FloatingLeaves />

      {/* Hero Header */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary via-primary/90 to-emerald-800">
          <div className="absolute inset-0 bg-gradient-to-b from-foreground/80 via-foreground/60 to-background" />
          <motion.div
            className="absolute inset-0 bg-gradient-to-tr from-primary/20 via-transparent to-accent/10"
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 6, repeat: Infinity }}
          />
        </div>

        <div className="relative mx-auto max-w-5xl w-full px-6 sm:px-8 pb-14 pt-16 sm:pt-24 sm:pb-20">
          {/* Decorative icons */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.15 }}
            className="absolute top-8 right-8 hidden sm:block"
          >
            <Sprout className="h-24 w-24 text-primary-foreground" />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="inline-flex items-center gap-2 rounded-full bg-primary-foreground/10 backdrop-blur-md border border-primary-foreground/15 px-4 py-2 text-xs font-semibold font-subtitle text-primary-foreground shadow-lg mb-6"
          >
            <HeartPulse className="h-3.5 w-3.5 text-accent" />
            LeafSense
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-heading font-normal text-primary-foreground leading-[1.08] mb-5"
          >
            Detect Plant
            <br />
            Diseases{" "}
            <span className="text-accent">Instantly</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="text-base sm:text-lg font-subtitle text-primary-foreground/85 max-w-xl leading-relaxed"
          >
            Upload a photo of your plant leaf and get an instant analysis
            for diseases, nutrient deficiencies, and overall health — all in seconds.
          </motion.p>

          {/* CTA hint */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-8 flex items-center gap-3"
          >
            <div className="flex -space-x-1">
              {["bg-primary", "bg-success", "bg-accent"].map((c, i) => (
                <div key={i} className={`w-8 h-8 rounded-full ${c} border-2 border-foreground/20 flex items-center justify-center`}>
                  <Leaf className="h-3.5 w-3.5 text-primary-foreground" />
                </div>
              ))}
            </div>
            <p className="text-sm font-subtitle text-primary-foreground/70">
              <span className="font-semibold text-primary-foreground/90">10,000+</span> plants scanned successfully
            </p>
          </motion.div>
        </div>

        {/* Curved bottom */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 50" fill="none" className="w-full h-auto">
            <path d="M0 50V25C360 0 720 0 1080 25C1260 37 1440 50 1440 50H0Z" fill="hsl(var(--background))" />
          </svg>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 mx-auto max-w-5xl w-full px-6 sm:px-8">
        <div className="absolute -top-24 left-1/2 -translate-x-1/2 h-48 w-96 bg-primary/6 rounded-full blur-3xl pointer-events-none" />

        <div className="relative space-y-8 pb-20">
          {/* Stats bar - pulled up */}
          <div className="-mt-6">
            <StatsBar />
          </div>

          {/* Feature cards - show when no file */}
          <AnimatePresence>
            {!file && !result && (
              <motion.div
                key="intro"
                exit={{ opacity: 0, y: -20, transition: { duration: 0.3 } }}
              >
                <FeatureCards />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Section label */}
          <div className="text-center space-y-1">
            <p className="text-xs uppercase tracking-[0.2em] font-semibold font-subtitle text-primary">
              Get Started
            </p>
            <h2 className="text-xl sm:text-2xl font-heading font-normal text-foreground">
              Upload Your Plant Image
            </h2>
            <div className="section-divider mt-1" />
          </div>

          {/* Upload section */}
          <section aria-label="Image upload">
            <ImageUpload
              onFileSelect={handleFileSelect}
              disabled={loading}
              previewUrl={previewUrl}
              onClear={handleClear}
            />
          </section>

          {/* Analyze button */}
          <AnimatePresence>
            {file && !result && !loading && (
              <motion.button
                key="analyze-btn"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                onClick={handleAnalyze}
                className="w-full flex items-center justify-center gap-2.5 rounded-2xl gradient-hero px-6 py-5 text-base font-bold text-primary-foreground shadow-elevated transition-all hover:shadow-glow hover:scale-[1.015] active:scale-[0.985] group"
              >
                <ScanLine className="h-5 w-5 group-hover:animate-pulse" />
                <span>Analyze Plant Health</span>
                <ArrowRight className="h-4 w-4 ml-1 group-hover:translate-x-1 transition-transform" />
              </motion.button>
            )}
          </AnimatePresence>

          {/* Loading state */}
          <AnimatePresence>
            {loading && (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <AnalyzingOverlay />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                key="error"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="rounded-2xl border border-destructive/30 bg-destructive/5 p-5"
              >
                <div className="flex items-start gap-3">
                  <div className="rounded-lg bg-destructive/15 p-2 flex-shrink-0">
                    <span className="text-destructive text-base">✕</span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-destructive">Prediction Failed</p>
                    <p className="mt-1 text-xs text-destructive/75 leading-relaxed">{error}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Result */}
          <AnimatePresence>
            {result && (
              <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                <ResultCard result={result} />
                <motion.button
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8 }}
                  onClick={handleClear}
                  whileHover={{ scale: 1.015 }}
                  whileTap={{ scale: 0.985 }}
                  className="w-full flex items-center justify-center gap-2 rounded-2xl glass-card px-6 py-4 text-sm font-bold text-foreground"
                >
                  <RotateCcw className="h-4 w-4" />
                  Scan Another Plant
                </motion.button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* How it works */}
          <HowItWorks />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/30 bg-muted/30 py-10">
        <div className="mx-auto max-w-5xl w-full px-6 sm:px-8 text-center space-y-4">
          <div className="flex items-center justify-center gap-2">
            <div className="rounded-lg gradient-hero p-1.5">
              <Leaf className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-heading text-lg font-normal text-foreground">LeafSense</span>
          </div>
          <p className="text-sm font-subtitle text-muted-foreground leading-relaxed max-w-md mx-auto">
            Built with deep learning for accurate plant disease detection.
            Helping farmers and gardeners protect their crops.
          </p>
          <div className="flex items-center justify-center gap-4 text-xs font-subtitle text-muted-foreground/70">
            <span>Disease Detection</span>
            <span>·</span>
            <span>Health Analysis</span>
            <span>·</span>
            <span>Nutrient Scoring</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
