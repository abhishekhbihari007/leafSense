import { useCallback, useState, useRef } from "react";
import { Upload, X, Camera, Image } from "lucide-react";

interface ImageUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  previewUrl: string | null;
  onClear: () => void;
}

export function ImageUpload({ onFileSelect, disabled, previewUrl, onClear }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith("image/")) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  if (previewUrl) {
    return (
      <div className="relative group animate-scale-in">
        <div className="overflow-hidden rounded-2xl glass-card">
          <div className="relative">
            <img
              src={previewUrl}
              alt="Selected plant image"
              className="w-full max-h-[400px] object-contain bg-muted/30 p-2"
            />
            {/* Overlay gradient */}
            <div className="absolute inset-0 bg-gradient-to-t from-foreground/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          </div>

          {/* Image info bar */}
          <div className="flex items-center justify-between px-5 py-3 border-t border-border/50 bg-muted/30">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Image className="h-4 w-4" />
              <span>Image ready for analysis</span>
            </div>
            <button
              onClick={onClear}
              disabled={disabled}
              className="flex items-center gap-1.5 rounded-lg bg-destructive/10 px-3 py-1.5 text-xs font-medium text-destructive hover:bg-destructive/20 transition-colors disabled:opacity-50"
              aria-label="Remove image"
            >
              <X className="h-3.5 w-3.5" />
              Remove
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`relative flex flex-col items-center justify-center gap-5 rounded-2xl border-2 border-dashed p-12 cursor-pointer transition-all duration-300 overflow-hidden group ${
          isDragging
            ? "border-primary bg-primary/5 scale-[1.01] shadow-glow"
            : "border-border/60 hover:border-primary/50 hover:bg-card/60"
        } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
      >
        {/* Background glow effect */}
        <div className="absolute inset-0 gradient-glow opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

        {/* Icon */}
        <div className="relative">
          <div className="rounded-2xl gradient-hero p-5 shadow-elevated group-hover:animate-float transition-all duration-300">
            <Camera className="h-8 w-8 text-primary-foreground" />
          </div>
          {/* Decorative ring */}
          <div className="absolute -inset-2 rounded-3xl border-2 border-primary/10 group-hover:border-primary/25 transition-colors duration-300" />
        </div>

        {/* Text */}
        <div className="relative text-center space-y-2">
          <p className="text-lg font-semibold text-foreground">
            Drop your plant photo here
          </p>
          <p className="text-sm text-muted-foreground">
            or <span className="text-primary font-medium underline underline-offset-2">browse files</span> from your device
          </p>
          <div className="flex items-center justify-center gap-2 pt-2">
            {["JPG", "PNG", "WEBP"].map((fmt) => (
              <span
                key={fmt}
                className="rounded-md bg-secondary px-2.5 py-1 text-[11px] font-medium text-secondary-foreground"
              >
                {fmt}
              </span>
            ))}
          </div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          name="image"
          onChange={handleChange}
          className="sr-only"
          disabled={disabled}
          aria-label="Upload plant image"
        />
      </label>
    </div>
  );
}
