import clsx from "clsx";
import { ChatMessage } from "./chatHistory";
import { useAudioVisualizerCircle } from "./useAudioVisualizerCircle";
import { useEffect, useRef } from "react";

const PositionedAudioVisualizer = ({
  chatHistory,
  role,
  analyserNode,
  isConnected,
  onCircleClick,
}: {
  chatHistory: ChatMessage[];
  role: "user" | "assistant";
  analyserNode: AnalyserNode | null;
  isConnected: boolean;
  onCircleClick?: () => void;
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isAssistant = role === "assistant";

  useAudioVisualizerCircle(canvasRef, {
    chatHistory,
    role,
    analyserNode,
    isConnected,
    showPlayButton: !!onCircleClick,
    clearCanvas: true,
  });

  // Resize the canvas to fit its parent element
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const size = Math.min(parent.clientWidth, parent.clientHeight);

    // If we don't do this `if` check, the recording ends up with flickering
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
    }
  });

  return (
    <div
      className={clsx(
        "max-w-3xl md:h-full flex items-center -mx-8 -my-8 px-4 md:px-0",
        isAssistant
          ? "md:w-full flex-row md:flex-row-reverse pt-36 md:pt-0"
          : "w-full flex-row-reverse md:flex-row md:pt-36 -ml-40 md:ml-0"
      )}
    >
      <div
        className={clsx(
          isAssistant ? "w-40 md:w-72 2xl:w-96" : "w-full md:w-48 2xl:w-72"
        )}
      >
        <canvas
          ref={canvasRef}
          className={`w-full h-full rounded-full ${
            onCircleClick ? "cursor-pointer" : ""
          }`}
          onClick={onCircleClick}
        />
      </div>
    </div>
  );
};

export default PositionedAudioVisualizer;
