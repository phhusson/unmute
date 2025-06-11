import clsx from "clsx";
import UnmuteHeader from "./UnmuteHeader";

export type HealthStatus = {
  connected: "no" | "yes_request_ok" | "yes_request_fail";
  ok: boolean;
  tts_up?: boolean;
  stt_up?: boolean;
  llm_up?: boolean;
  voice_cloning_up?: boolean;
};

const renderServiceStatus = (
  name: string,
  status: string | boolean | undefined,
  necessary: boolean = true
) => {
  if (status === undefined) {
    status = "Unknown";
  } else if (status === true) {
    status = "Up";
  } else if (status === false) {
    status = "Down";
  }

  return (
    <p>
      {name}:{" "}
      <span
        className={clsx(
          status === "Up"
            ? "text-white"
            : necessary
            ? "text-red"
            : "text-lightgray"
        )}
      >
        {status}
      </span>
    </p>
  );
};

const humanReadableStatus = {
  no: "Down",
  yes_request_ok: "Up",
  yes_request_fail: "Up, but with errors",
};

const CouldNotConnect = ({ healthStatus }: { healthStatus: HealthStatus }) => {
  if (healthStatus.ok) {
    return null;
  }

  return (
    <div className="w-full h-full flex flex-col gap-12 items-center justify-center bg-background">
      <UnmuteHeader />
      <div className="text-center text-xl">
        <h1 className="text-3xl mb-4">{"Couldn't connect :("}</h1>
        <p>Service status:</p>
        {renderServiceStatus(
          "Backend",
          humanReadableStatus[healthStatus.connected]
        )}
        {renderServiceStatus("STT", healthStatus.stt_up)}
        {renderServiceStatus("LLM", healthStatus.llm_up)}
        {renderServiceStatus("TTS", healthStatus.tts_up)}
        {renderServiceStatus(
          "Voice cloning",
          healthStatus.voice_cloning_up,
          false
        )}
      </div>
    </div>
  );
};

export default CouldNotConnect;
