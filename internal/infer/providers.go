package infer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/monbooru/monworker/internal/models"
	"github.com/monbooru/monworker/internal/shared/logx"
	ort "github.com/yalue/onnxruntime_go"
)

func directmlDeviceID() int {
	return parseDirectMLDeviceID(os.Getenv("MONWORKER_DIRECTML_DEVICE_ID"))
}

// The caller must run a non-nil cleanup to free the provider options. An
// empty cacheDir skips the compile cache.
func appendExecutionProvider(opts *ort.SessionOptions, provider, cacheDir string) (cleanup func(), err error) {
	switch provider {
	case "cpu", "":
		return nil, nil
	case "cuda":
		// Without this ORT keeps a host copy of the weights for the
		// session's lifetime.
		if err := opts.AddSessionConfigEntry("session.use_device_allocator_for_initializers", "1"); err != nil {
			return nil, fmt.Errorf("ort session config: %w", err)
		}
		// The device arena never shrinks, and the pattern's block would sit
		// in it beside the first run's chunks, doubling the VRAM held.
		if err := opts.SetMemPattern(false); err != nil {
			return nil, fmt.Errorf("set cuda mem pattern: %w", err)
		}
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("ort cuda options (ensure libonnxruntime was built with CUDA): %w", err)
		}
		// HEURISTIC skips cuDNN's multi-second exhaustive search,
		// kSameAsRequested grows the arena by the request instead of
		// doubling, and default-stream copies avoid a cross-stream sync.
		if err := cudaOpts.Update(map[string]string{
			"cudnn_conv_algo_search":    "HEURISTIC",
			"arena_extend_strategy":     "kSameAsRequested",
			"do_copy_in_default_stream": "1",
		}); err != nil {
			_ = cudaOpts.Destroy()
			return nil, fmt.Errorf("update cuda options: %w", err)
		}
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			_ = cudaOpts.Destroy()
			return nil, fmt.Errorf("append cuda provider: %w", err)
		}
		return func() { _ = cudaOpts.Destroy() }, nil
	case "directml":
		// DirectML requires the memory pattern off and sequential execution.
		if err := opts.SetMemPattern(false); err != nil {
			return nil, fmt.Errorf("set directml mem pattern: %w", err)
		}
		if err := opts.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
			return nil, fmt.Errorf("set directml execution mode: %w", err)
		}
		if err := opts.AppendExecutionProviderDirectML(directmlDeviceID()); err != nil {
			return nil, fmt.Errorf("append directml provider: %w", err)
		}
		return nil, nil
	case "tensorrt":
		trtOpts, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("ort tensorrt options (ensure libonnxruntime was built with TensorRT): %w", err)
		}
		if err := trtOpts.Update(map[string]string{}); err != nil {
			_ = trtOpts.Destroy()
			return nil, fmt.Errorf("update tensorrt options: %w", err)
		}
		if err := opts.AppendExecutionProviderTensorRT(trtOpts); err != nil {
			_ = trtOpts.Destroy()
			return nil, fmt.Errorf("append tensorrt provider: %w", err)
		}
		return func() { _ = trtOpts.Destroy() }, nil
	case "openvino":
		openvinoOpts := map[string]string{
			"device_type": "GPU",
		}
		if cacheDir != "" {
			loadCfg := map[string]any{
				"GPU": map[string]string{
					"CACHE_DIR":           cacheDir,
					"EXECUTION_MODE_HINT": "ACCURACY",
					"PERFORMANCE_HINT":    "LATENCY",
				},
			}
			loadCfgJSON, err := json.Marshal(loadCfg)
			if err != nil {
				return nil, fmt.Errorf("marshal openvino load_config: %w", err)
			}
			openvinoOpts["load_config"] = string(loadCfgJSON)
		}
		if err := opts.AppendExecutionProviderOpenVINO(openvinoOpts); err != nil {
			return nil, fmt.Errorf("append openvino provider: %w", err)
		}
		return nil, nil
	case "coreml":
		if err := opts.AppendExecutionProviderCoreML(0); err != nil {
			return nil, fmt.Errorf("append coreml provider: %w", err)
		}
		return nil, nil
	case "coremlv2":
		if err := opts.AppendExecutionProviderCoreMLV2(map[string]string{}); err != nil {
			return nil, fmt.Errorf("append coremlv2 provider: %w", err)
		}
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported execution provider %q", provider)
	}
}

// newDirectMLSession clears allowzero on the Reshapes the EP would mis-size,
// which reads the same since their shape holds no 0, and builds the session
// from that copy; the weights still load from the model's folder.
func newDirectMLSession(path string, inputs, outputs []string, opts *ort.SessionOptions) (*ort.DynamicAdvancedSession, error) {
	offsets, err := models.AllowZeroOffsets(path)
	if err != nil || len(offsets) == 0 {
		return ort.NewDynamicAdvancedSession(path, inputs, outputs, opts)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	for _, off := range offsets {
		data[off] = 0
	}
	if err := opts.AddSessionConfigEntry("session.model_external_initializers_file_folder_path", filepath.Dir(path)); err != nil {
		return nil, fmt.Errorf("ort session config: %w", err)
	}
	logx.Infof("infer: %s: allowzero cleared on %d Reshape nodes for directml", filepath.Base(filepath.Dir(path)), len(offsets))
	return ort.NewDynamicAdvancedSessionWithONNXData(data, inputs, outputs, opts)
}
