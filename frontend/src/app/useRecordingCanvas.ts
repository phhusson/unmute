import { useRef, useEffect, useCallback, useState } from "react";
import { AudioProcessor } from "./useAudioProcessor";
import { useAudioVisualizerCircle } from "./useAudioVisualizerCircle";
import { ChatMessage } from "./chatHistory";
import { getCSSVariable } from "./cssUtil";
import kyutaiLogo from "../assets/kyutai-logo-cropped.svg";

const getFilename = () => {
  const now = new Date();
  const pad = (n: number) => n.toString().padStart(2, "0");
  const dateStr = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(
    now.getDate()
  )}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
  return `unmute-${dateStr}.webm`;
};

export function useRecordingCanvas({
  size,
  shouldRecord,
  audioProcessor,
  chatHistory,
}: {
  size: number;
  shouldRecord: boolean;
  audioProcessor: AudioProcessor | null;
  chatHistory: ChatMessage[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>(0);

  // Recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);

  const logoImageRef = useRef<HTMLImageElement | null>(null);
  const [logoLoaded, setLogoLoaded] = useState(false);

  useEffect(() => {
    const img = new window.Image();
    img.src = kyutaiLogo.src;
    img.onload = () => setLogoLoaded(true);
    logoImageRef.current = img;
  }, []);

  useAudioVisualizerCircle(canvasRef, {
    chatHistory,
    role: "assistant",
    analyserNode: audioProcessor?.outputAnalyser || null,
    isConnected: true,
    showPlayButton: false,
    positioning: {
      centerX: size * 0.33,
      centerY: size * 0.38,
      radius: size * 0.3,
    },
    clearCanvas: false,
  });

  useAudioVisualizerCircle(canvasRef, {
    chatHistory,
    role: "user",
    analyserNode: audioProcessor?.inputAnalyser || null,
    isConnected: true,
    showPlayButton: false,
    positioning: {
      centerX: size * 0.7,
      centerY: size * 0.67,
      radius: size * 0.2,
    },
    clearCanvas: false,
  });

  const downloadRecording = useCallback(
    (asNewTab: boolean) => {
      if (!shouldRecord) {
        if (recordedChunks.length === 0) {
          throw new Error("No recording available to download.");
        }

        const blob = new Blob(recordedChunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        if (asNewTab) {
          window.open(url, "_blank");
        } else {
          const filename = getFilename();

          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          }, 100);
        }

        setRecordedChunks([]);
      }
    },
    [recordedChunks, shouldRecord]
  );

  const animate = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = getCSSVariable("--color-background");
    ctx.fillRect(0, 0, size, size);

    ctx.font = "bold 80px 'Frank Ruhl Libre'";
    ctx.fillStyle = "white";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Unmute.sh", size * 0.7, size * 0.1);

    // Draw Kyutai logo underneath the text
    if (logoLoaded && logoImageRef.current) {
      // Position: center under the text, scale to fit nicely
      const logoWidth = size * 0.25;
      const logoHeight =
        (logoImageRef.current.height / logoImageRef.current.width) * logoWidth;
      const logoX = size * 0.745 - logoWidth / 2;
      const logoY = size * 0.1 + 50; // 50px below the text, adjust as needed
      ctx.drawImage(logoImageRef.current, logoX, logoY, logoWidth, logoHeight);
    }

    animationFrameRef.current = requestAnimationFrame(animate);
  }, [size, logoLoaded]);

  // Initialize canvas and start animation
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");

    if (!canvas || !ctx) return;

    if (shouldRecord && audioProcessor) {
      const mediaStreamCombined = new MediaStream([
        ...canvas.captureStream(30).getTracks(),
        // ...assistantCanvasRef.current.captureStream(30).getTracks(),
        ...audioProcessor.mediaStreamDestination.stream.getAudioTracks(),
      ]);

      if (!mediaStreamCombined) {
        console.error("No media stream available");
        return;
      }

      // Supported MIME types are different for each browser
      let mimeType = "";
      const mimeTypesToTry = [
        "video/webm;codecs=vp8,opus", // Needed for Firefox
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
      ];
      for (const codec of mimeTypesToTry) {
        if (MediaRecorder.isTypeSupported(codec)) {
          mimeType = codec;
          break;
        }
      }

      mediaRecorderRef.current = new MediaRecorder(
        mediaStreamCombined,
        mimeType ? { mimeType } : undefined
      );

      setRecordedChunks([]);
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((chunks) => [...chunks, event.data]);
        }
      };
      mediaRecorderRef.current.start();
    } else {
      mediaRecorderRef.current?.stop();
    }

    // Set canvas size
    canvas.width = size;
    canvas.height = size;

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [
    size,
    // It's important to have shouldRecord here because when the hook is first
    // declared, the canvasRef is not available yet. We need to call this useEffect
    // later somehow, and shouldRecord is a good way to do that.
    shouldRecord,
    audioProcessor,
    animate,
  ]);

  return {
    canvasRef,
    downloadRecording,
    recordingAvailable: recordedChunks.length > 0,
  };
}
