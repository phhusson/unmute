export const base64EncodeOpus = (opusData: Uint8Array) => {
  // Convert to base64
  let binary = "";
  for (let i = 0; i < opusData.byteLength; i++) {
    binary += String.fromCharCode(opusData[i]);
  }
  return window.btoa(binary);
};

export const base64DecodeOpus = (base64String: string): Uint8Array => {
  const binaryString = window.atob(base64String);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};

export const convertWebmToWav = async (webmBlob: Blob): Promise<Blob> => {
  const arrayBuffer = await webmBlob.arrayBuffer();
  const AudioContextClass =
    window.AudioContext ||
    (window.hasOwnProperty("webkitAudioContext")
      ? (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext
      : undefined);
  if (!AudioContextClass) throw new Error("Web Audio API not supported");
  const audioCtx = new AudioContextClass();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  // Encode to wav
  const wavBuffer = encodeWAV(audioBuffer);
  return new Blob([wavBuffer], { type: "audio/wav" });
};

// Helper: Encode AudioBuffer to WAV format
export const encodeWAV = (audioBuffer: AudioBuffer): ArrayBuffer => {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  const samples = audioBuffer.length * numChannels;
  const buffer = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buffer);

  // Write WAV header
  function writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }
  let offset = 0;
  writeString(view, offset, "RIFF");
  offset += 4;
  view.setUint32(offset, 36 + samples * 2, true);
  offset += 4;
  writeString(view, offset, "WAVE");
  offset += 4;
  writeString(view, offset, "fmt ");
  offset += 4;
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, format, true);
  offset += 2;
  view.setUint16(offset, numChannels, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, (sampleRate * numChannels * bitDepth) / 8, true);
  offset += 4;
  view.setUint16(offset, (numChannels * bitDepth) / 8, true);
  offset += 2;
  view.setUint16(offset, bitDepth, true);
  offset += 2;
  writeString(view, offset, "data");
  offset += 4;
  view.setUint32(offset, samples * 2, true);
  offset += 4;

  // Write PCM samples
  for (let ch = 0; ch < numChannels; ch++) {
    const channel = audioBuffer.getChannelData(ch);
    for (let i = 0; i < channel.length; i++) {
      const sample = Math.max(-1, Math.min(1, channel[i]));
      view.setInt16(
        44 + (i * numChannels + ch) * 2,
        sample < 0 ? sample * 0x8000 : sample * 0x7fff,
        true
      );
    }
  }
  return buffer;
};
