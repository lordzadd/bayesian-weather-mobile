/**
 * iOS Metal-backed ExecuTorch wrapper for BMA inference.
 *
 * Mirrors the Android bma_executorch.cpp ABI exactly so the same
 * dart:ffi bindings in BmaEngine work on both platforms.
 *
 * Metal delegation is configured at export time via the MPS backend
 * in export_to_pte.py. ExecuTorch routes eligible ops to the GPU
 * automatically; CPU fallback is transparent.
 */

#import <Foundation/Foundation.h>
#include <cstring>
#include <memory>

// ExecuTorch headers — linked via Podfile CocoaPod
#include <executorch/runtime/platform/runtime.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

static constexpr int N_FEATURES  = 6;
static constexpr int SPATIAL_DIM = 2;

using namespace executorch::extension;

struct BmaModule {
    std::unique_ptr<Module> module;
};

extern "C" {

void* bma_load(const char* model_path) {
    executorch::runtime::runtime_init();

    auto* handle = new BmaModule();
    handle->module = std::make_unique<Module>(
        model_path,
        Module::LoadMode::MmapUseMlock
    );

    if (handle->module->load() != executorch::runtime::Error::Ok) {
        NSLog(@"[BmaExecutorch] Failed to load model: %s", model_path);
        delete handle;
        return nullptr;
    }

    NSLog(@"[BmaExecutorch] Model loaded: %s", model_path);
    return static_cast<void*>(handle);
}

void bma_infer(
    void*        handle,
    const float* gfs_input,
    const float* spatial_input,
    float*       out_mean,
    float*       out_std
) {
    if (!handle) return;
    auto* bma = static_cast<BmaModule*>(handle);

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
        NSLog(@"[BmaExecutorch] Inference failed");
        memset(out_mean, 0, N_FEATURES * sizeof(float));
        memset(out_std,  0, N_FEATURES * sizeof(float));
        return;
    }

    auto outputs = result.get();
    if (outputs.size() < 2) return;

    memcpy(out_mean, outputs[0].toTensor().const_data_ptr<float>(), N_FEATURES * sizeof(float));
    memcpy(out_std,  outputs[1].toTensor().const_data_ptr<float>(), N_FEATURES * sizeof(float));
}

void bma_free(void* handle) {
    if (!handle) return;
    delete static_cast<BmaModule*>(handle);
    NSLog(@"[BmaExecutorch] Module freed");
}

} // extern "C"
