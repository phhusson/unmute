import { useEffect, useRef } from "react";
import { LanguageCode, UnmuteConfig } from "./UnmuteConfigurator";
import { sendGAEvent } from "@next/third-parties/google";

interface ConversationAnalyticsInfo {
  voice: string;
  voice_name: string;
  is_custom_voice: boolean;
  instructions: string;
  instructions_type: string;
  instructions_language: LanguageCode;
  is_custom_instructions: boolean;
  start_timestamp_ms: number;
  conversation_uuid: string;
  duration_sec?: number;
}

export function useGoogleAnalytics({
  shouldConnect,
  unmuteConfig,
}: {
  shouldConnect: boolean;
  unmuteConfig: UnmuteConfig;
}) {
  const conversationAnalyticsInfo = useRef<ConversationAnalyticsInfo | null>(
    null
  );
  const unmuteConfigRef = useRef(unmuteConfig);

  // We keep the unmuteConfig in a ref because the useEffect that depends on it
  // should only run when shouldConnect changes, not when unmuteConfig changes.
  useEffect(() => {
    unmuteConfigRef.current = unmuteConfig;
  }, [unmuteConfig]);

  useEffect(() => {
    if (shouldConnect) {
      const config = unmuteConfigRef.current;
      const info = {
        voice: config.voice.startsWith("custom:") ? "custom" : config.voice,
        voice_name: config.voiceName,
        is_custom_voice: config.voice.startsWith("custom:"),
        instructions: JSON.stringify(config.instructions),
        instructions_language: config.instructions.language ?? "en",
        instructions_type: config.isCustomInstructions
          ? "constant_custom"
          : config.instructions.type,
        is_custom_instructions: config.isCustomInstructions,
        start_timestamp_ms: Date.now(),
        // Just to make it easy to pair with the end_conversation event
        conversation_uuid: crypto.randomUUID(),
      };
      conversationAnalyticsInfo.current = info;

      sendGAEvent("event", "start_conversation", info);
    } else {
      const info = conversationAnalyticsInfo.current;
      if (info) {
        info.duration_sec = (Date.now() - info.start_timestamp_ms) / 1000;
        sendGAEvent("event", "end_conversation", {
          ...info,
        });
      }
    }
  }, [shouldConnect]);

  const analyticsOnDownloadRecording = () => {
    const info = conversationAnalyticsInfo.current;
    if (info) {
      sendGAEvent("event", "download_recording", {
        ...info,
      });
    }
  };

  return { analyticsOnDownloadRecording };
}
