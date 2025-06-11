import { useRef, useEffect, useState } from "react";
import { getCSSVariable } from "./cssUtil";
import { ChatMessage } from "./chatHistory";

const WIDTH_INACTIVE = 3;
const WIDTH_ACTIVE = 5;

// Size scale factors for connected/disconnected states
const SCALE_CONNECTED = 1;
const SCALE_DISCONNECTED = 0.8;
const ANIMATION_DURATION = 500; // milliseconds

const INTERRUPTION_CHAR = "—"; // em-dash

const sampleToNormalizedRadius = (x: number) => {
  return 0.8 + 0.2 * Math.tanh(x * 2);
};

interface Positioning {
  centerX: number;
  centerY: number;
  radius: number;
}

const drawCircleVisualization = (
  canvas: HTMLCanvasElement,
  canvasCtx: CanvasRenderingContext2D,
  data: Float32Array,
  colorName: string,
  lineWidth: number,
  animationProgress: number,
  positioning: Positioning
) => {
  // Calculate scale factor from animation progress (0 = disconnected, 1 = connected)
  const scaleFactor =
    SCALE_DISCONNECTED +
    (SCALE_CONNECTED - SCALE_DISCONNECTED) * animationProgress;

  canvasCtx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const radius =
      positioning.radius * sampleToNormalizedRadius(data[i]) * scaleFactor;
    const angle = (i / data.length) * Math.PI * 2;

    const x = positioning.centerX + radius * Math.cos(angle);
    const y = positioning.centerY + radius * Math.sin(angle);
    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }
  }
  canvasCtx.closePath();
  canvasCtx.strokeStyle = getCSSVariable(colorName);
  canvasCtx.lineWidth = lineWidth;
  canvasCtx.stroke();
};

// New function to draw a play triangle
const drawPlayButton = (
  canvas: HTMLCanvasElement,
  canvasCtx: CanvasRenderingContext2D,
  colorName: string,
  animationProgress: number,
  positioning: Positioning
) => {
  // Calculate opacity based on animation progress (0 = fully visible, 1 = invisible)
  const opacity = 1 - animationProgress;

  if (opacity <= 0) return; // Don't draw if fully transparent

  const centerX = positioning.centerX;
  const centerY = positioning.centerY;
  const size = positioning.radius * 0.2; // Play button size relative to circle size

  // Create triangle path
  canvasCtx.beginPath();
  canvasCtx.moveTo(centerX + size / 2, centerY);
  canvasCtx.lineTo(centerX - size / 4, centerY - size / 2);
  canvasCtx.lineTo(centerX - size / 4, centerY + size / 2);
  canvasCtx.closePath();

  // Fill with color and opacity
  const color = getCSSVariable(colorName);
  // Parse the CSS variable color to get RGB values
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");
  if (!tempCtx) return;

  tempCtx.fillStyle = color;
  tempCtx.fillRect(0, 0, 1, 1);
  const rgba = tempCtx.getImageData(0, 0, 1, 1).data;

  // Apply opacity to the color
  canvasCtx.fillStyle = `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${opacity})`;
  canvasCtx.fill();
};

const getAnalyzerData = (
  analyserNode: AnalyserNode | null
): [Float32Array, Float32Array] => {
  const fftSize = 2048;
  const frequencyData = new Float32Array(fftSize / 2);
  const timeDomainData = new Float32Array(fftSize / 8);

  if (!analyserNode) {
    // return arrays corresponding to silence
    frequencyData.fill(-100); // -100 dBFS
    timeDomainData.fill(0); // silence

    return [frequencyData, timeDomainData];
  } else {
    // Configure analyzer node
    analyserNode.fftSize = fftSize;
    analyserNode.smoothingTimeConstant = 0.85;

    analyserNode.getFloatTimeDomainData(timeDomainData);
    analyserNode.getFloatFrequencyData(frequencyData);

    return [frequencyData, timeDomainData];
  }
};

const getIsActive = (
  chatHistory: ChatMessage[],
  role: "user" | "assistant"
) => {
  // Find the latest non-empty message from the specified role
  for (let i = chatHistory.length - 1; i >= 0; i--) {
    const message = chatHistory[i];

    // Empty messages, or ones where the LLM started generating but was interrupted
    // before it said anything
    if (message.content === "" || message.content === INTERRUPTION_CHAR)
      continue;

    if (message.content === "...") {
      // The user is silent, no more speech is coming
      return false;
    }

    if (message.role === role) {
      return true;
    } else {
      return false;
    }
  }
  // No non-empty messages found
  return false;
};

export interface UseAudioVisualizerCircleOptions {
  chatHistory: ChatMessage[];
  role: "user" | "assistant";
  analyserNode: AnalyserNode | null;
  isConnected?: boolean;
  showPlayButton?: boolean;
  positioning?: Positioning;
  clearCanvas: boolean;
}

export const useAudioVisualizerCircle = (
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  options: UseAudioVisualizerCircleOptions
) => {
  const {
    chatHistory,
    role,
    analyserNode,
    isConnected = false,
    showPlayButton = false,
    positioning,
    clearCanvas,
  } = options;

  const isActive = getIsActive(chatHistory, role);
  const isAssistant = role === "assistant";
  const colorName = isAssistant ? "color-green" : "color-white";

  const animationRef = useRef<number>(-1);
  const cicleBuffer = useRef<Float32Array>(new Float32Array(256));
  const circleIndex = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const animationStartTime = useRef<number | null>(null);
  const animationPreviousProgress = useRef<number>(isConnected ? 1 : 0);

  const interruptionTimeRef = useRef(0);
  const [interruptionIndex, setInterruptionIndex] = useState(0);

  // Single state for animation progress: 0 = disconnected state, 1 = connected state
  const [animationProgress, setAnimationProgress] = useState<number>(
    isConnected ? 1 : 0
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const stream = canvas.captureStream(30);
    stream.getTracks().forEach((track) => {
      track.stop();
    });
  }, [canvasRef]);

  useEffect(() => {
    if (chatHistory.length > interruptionIndex) {
      if (
        role === "user" &&
        chatHistory[chatHistory.length - 1].role === "assistant" &&
        // An interruption
        chatHistory[chatHistory.length - 1].content.endsWith(
          INTERRUPTION_CHAR
        ) &&
        // but not *only* an interruption char. That would mean the LLM got interrupted
        // before it said anything, and we don't want to count that as an interruption
        chatHistory[chatHistory.length - 1].content !== INTERRUPTION_CHAR
      ) {
        interruptionTimeRef.current = Date.now();
        setInterruptionIndex(chatHistory.length);
      }
    }
  }, [chatHistory, interruptionIndex, role]);

  // Handle connection state changes
  useEffect(() => {
    const animate = (timestamp: number) => {
      if (!animationStartTime.current) {
        animationStartTime.current = timestamp;
        animationPreviousProgress.current = animationProgress;
      }

      const elapsed = timestamp - animationStartTime.current;
      const duration = ANIMATION_DURATION;
      const progress = Math.min(elapsed / duration, 1);

      const targetProgress = isConnected ? 1 : 0;

      // Ease in-out function for smoother animation
      const easeInOutCubic = (t: number): number => {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      };

      const newProgress =
        animationPreviousProgress.current +
        easeInOutCubic(progress) *
          (targetProgress - animationPreviousProgress.current);

      setAnimationProgress(newProgress);

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        // Animation complete
        animationFrameRef.current = null;
        animationStartTime.current = null;
        setAnimationProgress(targetProgress); // Ensure we end exactly at target value
      }
    };

    // Cancel any existing animation
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Start the animation
    animationFrameRef.current = requestAnimationFrame(animate);

    // Cleanup on unmount
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isConnected, animationProgress]);

  // Main drawing effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const canvasCtx = canvas.getContext("2d");
    if (!canvasCtx) return;

    const draw = () => {
      // Calculate positioning - use provided positioning or default to center
      const currentPositioning: Positioning = positioning || {
        centerX: canvas.width / 2,
        centerY: canvas.height / 2,
        radius: Math.min(canvas.width / 2, canvas.height / 2),
      };

      const [frequencyData, timeDomainData] = getAnalyzerData(analyserNode);
      const volumeDbfs =
        frequencyData.reduce((a, b) => a + b, 0) / frequencyData.length;
      const volumeNormalized = (Math.max(-100, volumeDbfs) + 100) / 100;
      // cicleBuffer.current[circleIndex.current] = volumeNormalized;
      for (let i = 0; i < 1 + volumeNormalized * 10; i++) {
        cicleBuffer.current[circleIndex.current] = timeDomainData[i];
        circleIndex.current =
          (circleIndex.current + 1) % cicleBuffer.current.length;
      }

      // Schedule the next animation frame
      animationRef.current = requestAnimationFrame(draw);

      if (clearCanvas) {
        canvasCtx.clearRect(
          currentPositioning.centerX - currentPositioning.radius,
          currentPositioning.centerY - currentPositioning.radius,
          currentPositioning.radius * 2,
          currentPositioning.radius * 2
        );
      }

      const secSinceInterruption =
        (Date.now() - interruptionTimeRef.current) / 1000;
      const widthScale =
        1 + 2 * Math.exp(-Math.pow(secSinceInterruption * 3, 2));

      drawCircleVisualization(
        canvas,
        canvasCtx,
        cicleBuffer.current,
        colorName,
        Math.max(
          isActive ? WIDTH_ACTIVE : WIDTH_INACTIVE,
          WIDTH_INACTIVE * widthScale
        ),
        animationProgress,
        currentPositioning
      );

      // Draw play button if we have onClick and not fully connected
      if (showPlayButton && animationProgress < 1) {
        drawPlayButton(
          canvas,
          canvasCtx,
          colorName,
          animationProgress,
          currentPositioning
        );
      }
    };

    // Start the animation
    draw();

    // Clean up the animation on unmount
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [
    analyserNode,
    colorName,
    isActive,
    animationProgress,
    showPlayButton,
    canvasRef,
    positioning,
    clearCanvas,
  ]);

  return {};
};
