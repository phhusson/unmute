class MicInputProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.bufferSize = 960;
  }

  process(inputs, outputs, parameters) {
    // Get the input data from the first input's first channel
    const input = inputs[0][0];
    if (input && input.length > 0) {
      // Add the input data to the buffer
      this.buffer.push(...input);

      // Check if the buffer has reached the desired size
      if (this.buffer.length >= this.bufferSize) {
        // Post the buffered data to the main thread
        this.port.postMessage(this.buffer.slice(0, this.bufferSize));

        // Remove the sent samples from the buffer
        this.buffer = this.buffer.slice(this.bufferSize);
      }
    }

    // Return true to keep the processor alive
    return true;
  }
}

registerProcessor("mic-input-processor", MicInputProcessor);
