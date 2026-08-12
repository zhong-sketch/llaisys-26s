#include "context.hpp"
#include "../../utils.hpp"
#include <thread>

namespace llaisys::core {

Context::Context() : _current_runtime(nullptr) {
    // Keep the default path CPU-only. Other runtimes are created lazily by
    // setDevice(), so importing the library does not initialize CUDA.
    const LlaisysRuntimeAPI *api_ =
        llaisys::device::getRuntimeAPI(LLAISYS_DEVICE_CPU);
    const int device_count = api_->get_device_count();
    CHECK_ARGUMENT(device_count > 0, "CPU runtime is unavailable");

    auto &cpu_runtimes = _runtime_map[LLAISYS_DEVICE_CPU];
    cpu_runtimes.resize(device_count);
    cpu_runtimes[0] = new Runtime(LLAISYS_DEVICE_CPU, 0);
    cpu_runtimes[0]->_activate();
    _current_runtime = cpu_runtimes[0];
}

Context::~Context() {
    for (auto &runtime_entry : _runtime_map) {
        for (auto &runtime : runtime_entry.second) {
            if (runtime != nullptr) {
                if (runtime != _current_runtime) {
                    runtime->_activate();
                }
                delete runtime;
                runtime = nullptr;
            }
        }
    }
    _current_runtime = nullptr;
    _runtime_map.clear();
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    // If doest not match the current runtime.
    if (_current_runtime == nullptr || _current_runtime->deviceType() != device_type || _current_runtime->deviceId() != device_id) {
        auto &runtimes = _runtime_map[device_type];
        if (runtimes.empty()) {
            const auto *api = llaisys::device::getRuntimeAPI(device_type);
            const int device_count = api->get_device_count();
            CHECK_ARGUMENT(device_count > 0, "requested device type is unavailable");
            runtimes.resize(device_count);
        }
        CHECK_ARGUMENT(static_cast<size_t>(device_id) < runtimes.size() && device_id >= 0, "invalid device id");
        if (_current_runtime != nullptr) {
            _current_runtime->_deactivate();
        }
        if (runtimes[device_id] == nullptr) {
            runtimes[device_id] = new Runtime(device_type, device_id);
        }
        runtimes[device_id]->_activate();
        _current_runtime = runtimes[device_id];
    }
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

// Global API to get thread-local context.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core
