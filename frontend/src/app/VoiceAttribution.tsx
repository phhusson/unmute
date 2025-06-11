import Link from "next/link";
import { VoiceSample } from "./UnmuteConfigurator";

const VoiceAttribution = ({ voice }: { voice: VoiceSample }) => {
  const inner = () => {
    if (voice.source.source_type === "file") {
      if (voice.source.description_link) {
        return (
          <Link
            href={voice.source.description_link}
            className="underline"
            target="_blank"
            rel="noopener"
          >
            {voice.source.description ||
              "Source: " + voice.source.description_link}
          </Link>
        );
      } else if (voice.source.description) {
        return <>{voice.source.description}</>;
      } else {
        // No description or link provided
        return <></>;
      }
    } else {
      return (
        <>
          The &apos;{voice.name}&apos; voice is based on{" "}
          <Link
            href={voice.source.url}
            className="underline"
            target="_blank"
            rel="noopener"
          >
            this Freesound by {voice.source.sound_instance.username}
          </Link>
          .
        </>
      );
    }
  };
  return <div className="mt-2">{inner()}</div>;
};

export default VoiceAttribution;
