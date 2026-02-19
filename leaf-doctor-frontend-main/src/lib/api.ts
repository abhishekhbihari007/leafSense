// API configuration for the Plant Disease Detection backend
// Change this to your Flask API URL
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

export interface PredictionResult {
  class: "Healthy" | "Diseased" | string;
  confidence: number;
  message?: string;
  recommendation?: string;
  confidence_tier?: "high" | "moderate" | "low";
  nutrient_score?: number;
  random_confidence_score?: number;
}

export interface PredictionError {
  error: string;
}

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(data.error || `Prediction failed (status ${response.status})`);
  }

  return data as PredictionResult;
}
