// API configuration for the Plant Disease Detection backend
// Default to same-origin (\"\" -> relative \"/predict\") unless VITE_API_BASE_URL is set.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
const REQUEST_TIMEOUT_MS = 60_000; // 60 seconds

export interface PredictionResult {
  class: "Healthy" | "Diseased" | string;
  confidence: number;
  message?: string;
  recommendation?: string;
  confidence_tier?: "high" | "moderate" | "low";
  nutrient_score?: number | null;
}

export interface PredictionError {
  error: string;
}

export async function predictImage(file: File): Promise<PredictionResult> {
  if (file.size > MAX_IMAGE_SIZE_BYTES) {
    throw new Error("File is too large. Maximum size is 10 MB.");
  }

  const formData = new FormData();
  formData.append("image", file);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Request timed out. Please try again.");
    }
    throw err;
  }
  clearTimeout(timeoutId);

  if (response.ok === false && response.headers.get("content-type")?.includes("application/json") === false) {
    const fallback =
      response.status === 413
        ? "File too large. Maximum size is 10 MB."
        : response.status === 429
          ? "Too many requests. Please try again later."
          : `Request failed (${response.status}). Please try again.`;
    throw new Error(fallback);
  }

  let data: { error?: string } & PredictionResult;
  try {
    data = await response.json();
  } catch {
    const fallback =
      response.status === 413
        ? "File too large. Maximum size is 10 MB."
        : response.status === 429
          ? "Too many requests. Please try again later."
          : `Request failed (${response.status}). Please try again.`;
    throw new Error(fallback);
  }

  if (!response.ok || data.error) {
    throw new Error(data.error || `Prediction failed (status ${response.status})`);
  }

  return data as PredictionResult;
}
