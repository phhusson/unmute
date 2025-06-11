import React, { useEffect } from "react";
import { X } from "lucide-react";

export interface ErrorItem {
  id: string;
  message: string;
  timestamp: number;
}

export const makeErrorItem = (message: string): ErrorItem => {
  const timestamp = Date.now();
  return {
    id: `${timestamp}-${Math.random()}`,
    message,
    timestamp,
  };
};

const ERROR_TIMEOUT_SEC = 10;

export default function ErrorMessages({
  errors,
  setErrors,
}: {
  errors: ErrorItem[];
  setErrors: React.Dispatch<React.SetStateAction<ErrorItem[]>>;
}) {
  // Auto-dismiss errors after 10 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setErrors((prev) => {
        const now = Date.now();
        const filtered = prev.filter(
          (error) => now - error.timestamp < ERROR_TIMEOUT_SEC * 1000
        );
        return filtered;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [setErrors]);

  const handleDismiss = (index: number, errorId: string) => {
    setErrors((prev) => prev.filter((error) => error.id !== errorId));
  };

  if (errors.length === 0) {
    return null;
  }

  return (
    <div className="fixed top-4 left-0 md:left-4 z-50 space-y-2">
      {errors.map((error, index) => (
        <div
          key={error.id}
          className="bg-red-50 p-4 max-w-md"
          role="alert"
        >
          <div className="flex items-start gap-3">
            <div className="flex-1">
              <p className="text-red-800 text-sm font-medium">
                {error.message}
              </p>
            </div>
            <button
              onClick={() => handleDismiss(index, error.id)}
              className="flex-shrink-0 text-red-600 hover:text-red-800 transition-colors"
              aria-label="Dismiss error"
            >
              <X size={18} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
