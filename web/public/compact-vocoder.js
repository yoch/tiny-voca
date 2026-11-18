// Loaded by worker.js only when the compact pilot is requested.
const COMPACT_GEOMETRY = Object.freeze({
  sample_rate: 24000, hop: 256, synth_fft: 512, synth_hop: 128,
  noise_fft: 256, noise_hop: 64, cep_causal: 80, cep_anti: 16, noise_cep: 32,
});

function fillGaussian(target, random = Math.random) {
  // Box–Muller, two independent standard-normal samples per pair of draws.
  for (let i = 0; i < target.length; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    target[i] = radius * Math.cos(angle);
    if (i + 1 < target.length) target[i + 1] = radius * Math.sin(angle);
  }
}

class CompactVocoder {
  static async create(wasm, constants, maxFrames) {
    const { instance } = await WebAssembly.instantiate(wasm);
    return new CompactVocoder(instance.exports, constants, maxFrames);
  }

  constructor(abi, constants, maxFrames) {
    if (!Number.isSafeInteger(maxFrames) || maxFrames < 1 || maxFrames > 16384 ||
        constants.byteLength !== 896 * 4) throw Error("Invalid DSP configuration");
    this.abi = abi;
    this.maxFrames = maxFrames;
    this.framesCapacity = 0;
    abi._initialize();
    this.context = abi.__heap_base.value;
    this.contextBytes = abi.tv_context_size();
    this.arena = (this.context + this.contextBytes + 15) & ~15;
    this.ensureMemory(this.arena + constants.byteLength);
    this.heap.set(new Float32Array(constants.buffer, constants.byteOffset, 896), this.arena / 4);
    if (!abi.tv_initialize(this.context, this.contextBytes, this.arena, 896)) {
      throw Error("Invalid DSP constants");
    }
  }

  ensureMemory(end) {
    const memory = this.abi.memory;
    if (end > memory.buffer.byteLength) {
      memory.grow(Math.ceil((end - memory.buffer.byteLength) / 65536));
    }
    this.heap = new Float32Array(memory.buffer);
  }

  reserve(frames) {
    if (frames <= this.framesCapacity) return;
    // Round up geometrically, with a cap from the verified model configuration.
    const capacity = Math.min(this.maxFrames, Math.max(frames, 16, this.framesCapacity * 2));
    let next = this.arena / 4;
    this.offsets = [256, 98, 256, 36, 256, 256].map(width => {
      const offset = next;
      next += capacity * width;
      return offset;
    });
    this.workingBytes = (next - this.arena / 4) * 4;
    this.ensureMemory(next * 4);
    this.framesCapacity = capacity;
  }

  render(output, tokenCount, random = Math.random) {
    const { pulses, harmonic, noise_head: noiseHead, durations } = output;
    const frames = harmonic?.dims[1];
    if (!Number.isSafeInteger(frames) || frames < 1 || frames > this.maxFrames) {
      throw Error("Unsupported utterance size for compact DSP");
    }
    const samples = frames * 256;
    function hasShape(tensor, dims, type) {
      return tensor?.type === type && tensor.dims.length === dims.length &&
        dims.every((dim, index) => tensor.dims[index] === dim) &&
        tensor.data.length === dims.reduce((total, dim) => total * dim, 1);
    }
    if (!hasShape(pulses, [1, samples], "float32") ||
        !hasShape(harmonic, [1, frames, 98], "float32") ||
        !hasShape(noiseHead, [1, frames, 36], "float32") ||
        !hasShape(durations, [1, tokenCount], "int64") ||
        durations.data.some(value => value < 0n || value > BigInt(frames)) ||
        durations.data.reduce((total, value) => total + value, 0n) !== BigInt(frames)) {
      throw Error("Compact model output dimensions do not match");
    }
    this.reserve(frames);
    const [p, h, n, nh, a, scratch] = this.offsets;
    const heap = this.heap;
    heap.set(pulses.data, p);
    heap.set(harmonic.data, h);
    heap.set(noiseHead.data, nh);
    const started = performance.now();
    fillGaussian(heap.subarray(n, n + samples), random);
    const noiseTime = performance.now() - started;
    const dspStarted = performance.now();
    if (!this.abi.tv_render(this.context, this.contextBytes, frames,
      p * 4, samples, h * 4, frames * 98, n * 4, samples, nh * 4, frames * 36,
      a * 4, samples, scratch * 4, samples)) throw Error("Compact DSP rejected audio or buffers");
    const audio = heap.slice(a, a + samples);
    return { audio, noiseTime, dspTime: performance.now() - dspStarted,
      wasmBytes: heap.byteLength, workspaceBytes: this.workingBytes };
  }
}
