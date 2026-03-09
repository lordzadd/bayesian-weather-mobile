/**
 * Native C++ wrapper for ExecuTorch BMA model inference.
 *
 * Exposes a plain C API consumed by Flutter via dart:ffi:
 *   bma_load()  — loads .pte model into ExecuTorch Module
 *   bma_infer() — runs forward pass, returns posterior mean + std
 *   bma_free()  — releases module resources
 *
 * GPU delegation (Vulkan) is configured at export time in export_to_pte.py.
 * At runtime, ExecuTorch automatically routes delegated ops to Vulkan if
 * the device supports it; CPU fallback is transparent.
 */

#include <jni.h>
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

// ExecuTorch headers — available after adding the AAR dependency
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#define LOG_TAG "BmaExecutorch"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static constexpr int N_FEATURES = 6;
static constexpr int SPATIAL_DIM = 2;

using namespace executorch::extension;

struct BmaModule {
    std::unique_ptr<Module> module;
};

extern "C" {

/**
 * Load a .pte model from disk.
 * @param model_path  Absolute path to bma_model.pte in app documents dir.
 * @return            Opaque handle, or nullptr on failure.
 */
void* bma_load(const char* model_path) {
    executorch::runtime::runtime_init();

    auto* handle = new BmaModule();
    handle->module = std::make_unique<Module>(
        model_path,
        Module::LoadMode::MmapUseMlock
    );

    auto load_result = handle->module->load();
    if (load_result != executorch::runtime::Error::Ok) {
        LOGE("Failed to load model from %s", model_path);
        delete handle;
        return nullptr;
    }

    LOGI("BMA model loaded from %s", model_path);
    return static_cast<void*>(handle);
}

/**
 * Run posterior inference.
 *
 * @param handle       Opaque module handle from bma_load().
 * @param gfs_input    float[6]  — GFS forecast feature vector.
 * @param spatial_input float[2] — [lat, lon] normalized to [-1, 1].
 * @param out_mean     float[6]  — posterior mean output (caller-allocated).
 * @param out_std      float[6]  — posterior std dev output (caller-allocated).
 */
void bma_infer(
    void*        handle,
    const float* gfs_input,
    const float* spatial_input,
    float*       out_mean,
    float*       out_std
) {
    if (!handle) {
        LOGE("bma_infer called with null handle");
        return;
    }

    auto* bma = static_cast<BmaModule*>(handle);

    // Build input tensors
    auto gfs_tensor = from_blob(
        const_cast<float*>(gfs_input),
        {1, N_FEATURES},
        executorch::runtime::ScalarType::Float
    );
    auto spatial_tensor = from_blob(
        const_cast<float*>(spatial_input),
        {1, SPATIAL_DIM},
        executorch::runtime::ScalarType::Float
    );

    auto result = bma->module->forward({
        EValue(gfs_tensor.get()),
        EValue(spatial_tensor.get()),
    });

    if (!result.ok()) {
        LOGE("Inference failed");
        // Return zeros on failure — caller checks for NaN/zero
        memset(out_mean, 0, N_FEATURES * sizeof(float));
        memset(out_std,  0, N_FEATURES * sizeof(float));
        return;
    }

    // Unpack (mean, std) tuple from model output
    auto outputs = result.get();
    if (outputs.size() < 2) {
        LOGE("Expected 2 output tensors, got %zu", outputs.size());
        return;
    }

    const auto& mean_tensor = outputs[0].toTensor();
    const auto& std_tensor  = outputs[1].toTensor();

    const float* mean_data = mean_tensor.const_data_ptr<float>();
    const float* std_data  = std_tensor.const_data_ptr<float>();

    memcpy(out_mean, mean_data, N_FEATURES * sizeof(float));
    memcpy(out_std,  std_data,  N_FEATURES * sizeof(float));
}

/**
 * Release model resources.
 * @param handle  Opaque handle from bma_load().
 */
void bma_free(void* handle) {
    if (!handle) return;
    delete static_cast<BmaModule*>(handle);
    LOGI("BMA module freed");
}

} // extern "C"
