"use client";
import { useState } from "react";
import SlantedButton from "../SlantedButton";
import { useBackendServerUrl } from "../useBackendServerUrl";
import ErrorMessages, { ErrorItem, makeErrorItem } from "../ErrorMessages";
import VoiceRecording, { RecordedAudio } from "../VoiceRecorder";
import Link from "next/link";

type VoiceDonationVerification = {
  id: string;
  text: string;
  created_at_timestamp: number; // Seconds since epoch
};

export default function VoiceDonation() {
  const [errors, setErrors] = useState<ErrorItem[]>([]);
  const backendServerUrl = useBackendServerUrl();
  const [verification, setVerification] =
    useState<VoiceDonationVerification | null>(null);
  const [recordedAudio, setRecordedAudio] = useState<RecordedAudio | null>(
    null
  );
  const [consentGiven, setConsentGiven] = useState(false);
  const [nickname, setNickname] = useState("");

  const [uploadState, setUploadState] = useState<
    "not_started" | "uploading" | "finished"
  >("not_started");

  const addError = (error: string) => {
    setErrors((prev) => [...prev, makeErrorItem(error)]);
  };

  const onRecordingStarted = async () => {
    if (!backendServerUrl) return;

    try {
      // This doesn't actually exist on the backend yet
      const response = await fetch(`${backendServerUrl}/v1/voice-donation`);
      if (!response.ok) {
        throw new Error("Failed to get voice donation verification.");
      }
      const data = await response.json();
      setVerification(data);
    } catch (error) {
      // console.error("Error fetching voice donation verification:", error);
      setErrors((prev) => [
        ...prev,
        makeErrorItem(
          error instanceof Error
            ? error.message
            : "Failed to start voice donation."
        ),
      ]);
    }
  };

  const handleSubmit = async () => {
    if (!recordedAudio) {
      addError("You haven't recorded your voice yet.");
      return;
    }
    if (!verification) {
      addError("No active voice donation verification.");
      return;
    }
    if (!consentGiven) {
      addError("You must give consent to submit your voice.");
      return;
    }

    setUploadState("uploading");

    const formData = new FormData();
    formData.append("file", recordedAudio.file);

    const metadata = {
      nickname: nickname,
      verification_id: verification?.id || null,
    };
    formData.append("metadata", JSON.stringify(metadata));

    try {
      // This doesn't actually exist on the backend yet
      const response = await fetch(`${backendServerUrl}/v1/voice-donation`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        addError(error.detail);
        setUploadState("not_started");
        return;
      }

      const data = await response.json();
      console.log("Submit response:", data);
    } catch (err) {
      addError(
        err instanceof Error ? err.message : "An unknown error occurred."
      );
    }
    setUploadState("finished");
  };

  if (!backendServerUrl) {
    return (
      <div className="w-full h-screen flex justify-center items-center bg-background">
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="w-full min-h-screen flex justify-center bg-background">
      <ErrorMessages errors={errors} setErrors={setErrors} />
      <div className="flex flex-col justify-center max-w-xl gap-3 m-2">
        <h1 className="text-4xl font-bold mt-4">Voice Donation</h1>
        <p className="italic">
          <Link href="/" className="underline">
            Back to Unmute
          </Link>
        </p>
        {uploadState === "finished" && (
          <>
            <p>Thank you for donating your voice for open science &lt;3</p>
            {verification && (
              <>
                <p>
                  The identifier of your voice donation is{" "}
                  <span className="font-mono font-bold">{verification.id}</span>
                  .
                </p>
                <p>
                  Please save it now because it will not be shown again. You can
                  use this identifier to find your voice later.{" "}
                </p>
                <p>
                  <Link href={"/"} className="underline">
                    Go back to Unmute
                  </Link>
                </p>
              </>
            )}
          </>
        )}
        {uploadState !== "finished" && (
          <>
            <p>
              Here you can anonymously donate your voice to be used for voice
              cloning by Kyutai&apos;s{" "}
              <span className="text-green">
                open-science text-to-speech model
              </span>
              . We are looking for pre-made voices to include alongside the
              open-source release of the Kyutai TTS (coming soon).
            </p>
            <p>
              The voice embedding and/or voice sample will be stored by Kyutai
              and may be released anonymously under the{" "}
              <Link
                href="https://creativecommons.org/publicdomain/zero/1.0/"
                className="underline "
                target="_blank"
                rel="noopener"
              >
                CC0 license
              </Link>
              . We may also decide not to release the voice.{" "}
            </p>
            <p>
              <strong>
                Be aware that donating your voice means that users of our TTS
                will be able to use your voice to say anything.
              </strong>{" "}
              The voice recording is not linked to your personal information in
              any way, but of course, it can still be possible to recognize you
              by the voice.
            </p>
            <h2 className="text-xl strong mt-4">Verification</h2>
            <div className="w-full flex flex-row">
              <div className="border-1 border-green p-2 text-center">
                Verbal consent
              </div>
              <div className="border-1 border-green p-2 text-center">
                Verification sentences
              </div>
              <div className="border-1 border-green p-2 text-center grow-2">
                Whatever you want (last 10 seconds will be used)
              </div>
            </div>
            <p>
              To verify that this is your voice, we will ask you to read a short
              text out loud. Afterwards, you can say whatever you want. Have fun
              with it! The TTS is good at reproducing the tone and mannerisms of
              the voice. The last 10 seconds of your recording will be used as
              the voice sample. Try to use the same tone throughout the
              recording to make it easier to verify that it&apos;s you.
            </p>
            {!recordedAudio && (
              <p>
                You&apos;ll have the chance to listen to your recording before
                submitting it.
              </p>
            )}
            <VoiceRecording
              setRecordedAudio={setRecordedAudio}
              setError={(error: string | null) => {
                if (!error) return;
                addError(error);
              }}
              recordingDurationSec={30}
              onRecordingStarted={onRecordingStarted}
              showProgress={false}
            />
            {verification && (
              <div>
                <p>Start by saying:</p>
                <p className="italic">{verification.text}</p>
              </div>
            )}
            {recordedAudio && (
              <div className="flex flex-col gap-2">
                <label className="flex flex-col gap-1">
                  (Optional) Preferred nickname for the voice if published:
                  <input
                    type="text"
                    value={nickname}
                    onChange={(e) => setNickname(e.target.value)}
                    className="border px-2 py-1 bg-gray text-white"
                  />
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={consentGiven}
                    onChange={(e) => setConsentGiven(e.target.checked)}
                  />
                  <p>
                    I consent to my voice being used for voice cloning under the
                    terms described above. <span className="text-red">*</span>
                  </p>
                </label>
                <SlantedButton
                  kind={
                    consentGiven && uploadState === "not_started"
                      ? "primary"
                      : "disabled"
                  }
                  disabled={!consentGiven}
                  onClick={handleSubmit}
                >
                  {uploadState === "uploading" ? "Uploading..." : "Submit"}
                </SlantedButton>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
