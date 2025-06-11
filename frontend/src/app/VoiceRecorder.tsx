import { useRef, useState } from "react";
import SlantedButton from "./SlantedButton";
import { convertWebmToWav } from "./audioUtil";
import { Mic } from "lucide-react";
import clsx from "clsx";

export const MIC_RECORDING_FILENAME = "unmute-mic-recording.wav";

export type RecordedAudio = {
  blobUrl: string;
  file: File;
};

const VoiceRecording = ({
  setRecordedAudio,
  setError,
  recordingDurationSec,
  onRecordingStarted,
  showProgress = true,
}: {
  setRecordedAudio: (recordedAudio: RecordedAudio) => void;
  setError: (error: string | null) => void;
  recordingDurationSec: number;
  onRecordingStarted?: () => void;
  showProgress?: boolean;
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(
    null
  );
  const [recordingProgress, setRecordingProgress] = useState(0);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [recordedAudioLocal, setRecordedAudioLocal] =
    useState<RecordedAudio | null>(null);

  const handleStartRecording = async () => {
    setError(null);
    onRecordingStarted?.();
    setRecordingProgress(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Prefer audio/wav if supported. The backend can't handle webm, so we need to convert it.
      // If neither is supported, don't specify and hope for the best. (That seems to work on Safari.)
      let mimeType = "";
      if (MediaRecorder.isTypeSupported("audio/wav")) {
        mimeType = "audio/wav";
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        mimeType = "audio/webm";
      }

      const recorder = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      recorder.onstop = async () => {
        setRecordingProgress(0);
        if (recordingIntervalRef.current) {
          clearInterval(recordingIntervalRef.current);
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

        let audioFile: File;
        if (mimeType === "audio/wav") {
          audioFile = new File([audioBlob], MIC_RECORDING_FILENAME, {
            type: "audio/wav",
          });
        } else {
          const wavBlob = await convertWebmToWav(audioBlob);
          audioFile = new File([wavBlob], MIC_RECORDING_FILENAME, {
            type: "audio/wav",
          });
        }
        const recordedAudio: RecordedAudio = {
          blobUrl: URL.createObjectURL(audioFile),
          file: audioFile,
        };
        setRecordedAudio(recordedAudio);
        setRecordedAudioLocal(recordedAudio);
      };
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);

      const start = Date.now();
      recordingIntervalRef.current = setInterval(() => {
        const elapsed = (Date.now() - start) / 1000;
        setRecordingProgress(Math.min(elapsed / recordingDurationSec, 1));
      }, 50);

      setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
          setIsRecording(false);
          setMediaRecorder(null);
        }
      }, recordingDurationSec * 1000);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not access microphone."
      );
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setIsRecording(false);
      setMediaRecorder(null);
    }
    setRecordingProgress(0);
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
    }
  };

  return (
    <div>
      <div className="flex gap-2 items-center">
        <div className="w-full flex justify-center">
          <SlantedButton
            onClick={isRecording ? handleStopRecording : handleStartRecording}
            kind={
              isRecording || recordedAudioLocal != null
                ? "secondary"
                : "primary"
            }
            extraClasses="flex items-center gap-2"
          >
            {isRecording ? (
              "● Recording"
            ) : (
              <>
                <Mic size={24} />
                Record
              </>
            )}
          </SlantedButton>
        </div>
      </div>
      {showProgress && (
        <div
          className={clsx(
            "w-full h-2 overflow-hidden mt-2",
            isRecording ? "bg-lightgray" : "bg-transparent"
          )}
        >
          <div
            className="h-full bg-red transition-all duration-50"
            style={{ width: `${recordingProgress * 100}%` }}
          ></div>
        </div>
      )}
      {recordedAudioLocal && !isRecording && (
        <audio
          controls
          src={recordedAudioLocal.blobUrl}
          className="w-full mt-2"
        />
      )}
    </div>
  );
};

export default VoiceRecording;
