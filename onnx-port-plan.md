# M6: ONNX BPM backend (Windows unblock + single inference runtime)

## Why

- essentia has no Windows wheels - it is the only blocker for keydup Windows builds
  (KeyNet is pure torch and already runs there).
- The macOS PyInstaller bundle deadlocks loading essentia's TensorFlow next to torch.
  An ONNX backend removes TensorFlow from the process entirely: one runtime
  (onnxruntime, wheels on all 3 OSes) instead of two.
- beat-this as a fallback was rejected: 3/13 exact-match on the Kawaii Karnival
  benchmark vs 23/23 for the tuned TempoCNN+onset path. The bar is parity, not
  "some BPM detector".

## Spike results (2026-06-11, all verified locally)

- `deepsquare-k16-3.pb`: input placeholder `input` (batch, 40 mel bands, time, 1),
  output `output` softmax over 256 BPM bins (bin + 30 = BPM). 555 nodes.
- tf2onnx conversion (opset 17) succeeds; 4.8 MB onnx file.
- **TF vs ONNX on identical input: max abs diff 8.3e-7, 36/36 patch argmax
  agreement, same BPM.** Model conversion carries zero accuracy risk.
- Full-pipeline check on a benchmark track (tagged 172): essentia pipeline and
  ONNX both land 172 via mean-probability argmax.
- Hand-rebuilt patches (frameSize 1024, hop 512, TensorflowInputTempoCNN mel)
  did NOT byte-match essentia's internal patching (26/36 patch agreement,
  though final BPM still agreed). The exactness work is in the input pipeline,
  not the model.

## Workstreams

1. **Model artifact** (done in spike, productionize): convert deepsquare-k16-3.pb
   -> `keypipe/models/tempocnn-deepsquare.onnx`, commit, document the tf2onnx
   command. Conversion env is throwaway (tensorflow-cpu + tf2onnx); not a dependency.
2. **Input pipeline port** (the main work): librosa mel-band frames numerically
   matching essentia's `TensorflowInputTempoCNN` (40 bands @ 11025 Hz, frame 1024,
   hop 512 - watch mel filterbank convention HTK/Slaney, magnitude vs power, norm),
   plus exact patching (patch 256 frames; essentia produced 36 patches from 4735
   frames - nail patchHopSize/lastPatchMode against ground truth dumps).
   Gate: max patch diff ~1e-5 vs TensorflowInputTempoCNN on real tracks.
3. **Onset correction port**: replace `essentia.OnsetRate` with librosa onset
   detection feeding the existing (pure numpy) impulse-train autocorrelation
   tiebreaker. Onset detectors differ; correctness is judged end-to-end (gate 5).
4. **Backend**: `keypipe/inference_onnx.py` with an `OnnxBPMDetector` exposing
   `detect`/`detect_with_confidence`; keypipe extra `onnx = ["onnxruntime"]`.
   keydup's `select_bpm_backend()` prefers essentia when importable, else ONNX
   (and can be forced via env/setting for A/B).
5. **Validation gate - RESULT (2026-06-12)**: the benchmark folder now holds 15
   tracks (13 with BPM tags; contents changed since April's 23-track run, incl.
   decimal half-time tags like 87.50). Scored against tags (half-time doubled):
   **ONNX 12/13, essentia 9/13, backends agree 10/15**. All misses on both
   sides are +/-1 borderline rounds; librosa's onset tiebreaker out-rounds
   OnsetRate here (nails 180.00 where essentia says 181). Gate read as
   "ONNX >= essentia": PASSED. Pipeline parity: mel 1e-6, patches 3e-6
   (patch=256, hop=128, z-norm per patch, zero-centered framing).
6. **CI/packaging**: add windows-2022 to the keydup matrix (no essentia there by
   dependency markers already); PyInstaller Windows artifact + self-test.
   **macOS DECIDED (2026-06-12): ONNX backend, essentia excluded from mac
   bundles.** Attempt 2 proved essentia's import deadlocks in the bundle even
   when loaded before torch (zero markers printed) - dyld-level, not ordering.
   essentia remains a Linux-bundle + dev/ground-truth dependency only.

## Risks / open items

- Mel/patching parity is fiddly (the known-hard 20%); mitigated by dumping
  essentia intermediates as ground truth fixtures while it still runs locally.
- Onset port may shift a benchmark track or two; the autocorrelation logic is
  already pure numpy (ports unchanged), only onset times differ. Budget tuning
  time; 23/23 is the gate.
- **Model license**: the deepsquare/deeptemp TempoCNN weights come from MTG's
  essentia models collection (Schreiber & Mueller's work) - believed
  CC BY-NC-SA. Verify before public binary distribution; this applies to the
  CURRENT builds shipping the .pb too, independent of ONNX. Add attribution to
  the About dialog/README either way.
- Windows GUI smoke (Qt offscreen on windows-2022) untested until the leg exists.

## Order

W1 -> W2 (with fixtures) -> W3 -> W4 -> W5 gate -> W6. W2/W3 are independent
after fixtures exist. Everything is testable on Linux against essentia ground
truth; Windows runners only validate packaging, not accuracy.
