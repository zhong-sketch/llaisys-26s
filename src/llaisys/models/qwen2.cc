#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#pragma warning(disable : 4297)
#endif

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};

    std::vector<llaisys::tensor_t> kv_cache_k;
    std::vector<llaisys::tensor_t> kv_cache_v;
    size_t cache_capacity = 0;
    size_t total_kv_len = 0;

    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
};

namespace {

using llaisys::tensor_t;

tensor_t unwrap(llaisysTensor_t handle, const char *name) {
    if (handle == nullptr || handle->tensor == nullptr) {
        throw std::runtime_error(std::string("Qwen2 weight is not loaded: ") + name);
    }
    return handle->tensor;
}

tensor_t make_tensor(const std::vector<size_t> &shape,
                     llaisysDataType_t dtype,
                     llaisysDeviceType_t device,
                     int device_id) {
    return llaisys::Tensor::create(shape, dtype, device, device_id);
}

void copy_tensor(const tensor_t &dst, const tensor_t &src) {
    CHECK_ARGUMENT(dst != nullptr && src != nullptr, "Qwen2 copy requires valid tensors");
    CHECK_ARGUMENT(dst->numel() == src->numel(), "Qwen2 copy size mismatch");
    CHECK_ARGUMENT(dst->dtype() == src->dtype(), "Qwen2 copy dtype mismatch");
    CHECK_ARGUMENT(dst->deviceType() == src->deviceType() &&
                       dst->deviceId() == src->deviceId(),
                   "Qwen2 copy device mismatch");

    llaisys::core::context().setDevice(dst->deviceType(), dst->deviceId());
    const size_t bytes = dst->numel() * dst->elementSize();
    const auto kind = dst->deviceType() == LLAISYS_DEVICE_CPU
                          ? LLAISYS_MEMCPY_H2H
                          : LLAISYS_MEMCPY_D2D;
    llaisys::core::context().runtime().api()->memcpy_sync(
        dst->data(), src->data(), bytes, kind);
}

int64_t read_i64(const tensor_t &tensor) {
    int64_t value = 0;
    llaisys::core::context().setDevice(tensor->deviceType(), tensor->deviceId());
    const auto kind = tensor->deviceType() == LLAISYS_DEVICE_CPU
                          ? LLAISYS_MEMCPY_H2H
                          : LLAISYS_MEMCPY_D2H;
    llaisys::core::context().runtime().api()->memcpy_sync(
        &value, tensor->data(), sizeof(value), kind);
    return value;
}

void ensure_weights(const LlaisysQwen2Model *model) {
    if (model->weights.in_embed == nullptr ||
        model->weights.out_embed == nullptr ||
        model->weights.out_norm_w == nullptr) {
        throw std::runtime_error("Qwen2 global weights are incomplete");
    }

    for (size_t l = 0; l < model->meta.nlayer; l++) {
        if (model->weights.attn_norm_w[l] == nullptr ||
            model->weights.attn_q_w[l] == nullptr ||
            model->weights.attn_k_w[l] == nullptr ||
            model->weights.attn_v_w[l] == nullptr ||
            model->weights.attn_o_w[l] == nullptr ||
            model->weights.mlp_norm_w[l] == nullptr ||
            model->weights.mlp_gate_w[l] == nullptr ||
            model->weights.mlp_up_w[l] == nullptr ||
            model->weights.mlp_down_w[l] == nullptr) {
            throw std::runtime_error("Qwen2 layer weights are incomplete");
        }
    }
}

void ensure_kv_cache(LlaisysQwen2Model *model, size_t required_len) {
    CHECK_ARGUMENT(required_len <= model->meta.maxseq,
                   "Qwen2 sequence length exceeds maxseq");

    if (required_len <= model->cache_capacity) {
        return;
    }

    size_t next_capacity = std::max<size_t>(256, model->cache_capacity * 2);
    next_capacity = std::max(next_capacity, required_len);
    next_capacity = std::min(next_capacity, model->meta.maxseq);

    for (size_t l = 0; l < model->meta.nlayer; l++) {
        auto next_k = make_tensor(
            {next_capacity, model->meta.nkvh, model->meta.dh},
            model->meta.dtype, model->device, model->device_id);
        auto next_v = make_tensor(
            {next_capacity, model->meta.nkvh, model->meta.dh},
            model->meta.dtype, model->device, model->device_id);

        if (model->cache_capacity > 0) {
            auto old_k = model->kv_cache_k[l];
            auto old_v = model->kv_cache_v[l];
            const size_t old_numel =
                model->cache_capacity * model->meta.nkvh * model->meta.dh;
            auto old_k_view = old_k->slice(0, 0, model->cache_capacity);
            auto old_v_view = old_v->slice(0, 0, model->cache_capacity);
            auto next_k_view = next_k->slice(0, 0, model->cache_capacity);
            auto next_v_view = next_v->slice(0, 0, model->cache_capacity);
            CHECK_ARGUMENT(old_k_view->numel() == old_numel &&
                               old_v_view->numel() == old_numel,
                           "Qwen2 KV-Cache size mismatch");
            copy_tensor(next_k_view, old_k_view);
            copy_tensor(next_v_view, old_v_view);
        }

        model->kv_cache_k[l] = std::move(next_k);
        model->kv_cache_v[l] = std::move(next_v);
    }

    model->cache_capacity = next_capacity;
}

tensor_t add_residual(const tensor_t &value, const tensor_t &residual,
                      llaisysDeviceType_t device, int device_id) {
    auto out = make_tensor(value->shape(), value->dtype(), device, device_id);
    llaisys::ops::add(out, value, residual);
    return out;
}

} // namespace

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    if (meta == nullptr) {
        throw std::invalid_argument("Qwen2 meta must not be null");
    }

    auto *model = new LlaisysQwen2Model;
    model->meta = *meta;
    model->device = device;
    model->device_id = (device_ids != nullptr && ndevice > 0) ? device_ids[0] : 0;

    if (model->meta.dh == 0) {
        CHECK_ARGUMENT(model->meta.nh > 0 &&
                           model->meta.hs % model->meta.nh == 0,
                       "Qwen2 hidden size must divide attention heads");
        model->meta.dh = model->meta.hs / model->meta.nh;
    }
    CHECK_ARGUMENT(model->meta.maxseq > 0, "Qwen2 maxseq must be positive");
    CHECK_ARGUMENT(model->meta.nlayer > 0 &&
                       model->meta.nh > 0 &&
                       model->meta.nkvh > 0,
                   "Qwen2 meta contains invalid layer configuration");

    const size_t nlayer = model->meta.nlayer;
    model->weights.attn_norm_w = new llaisysTensor_t[nlayer]{};
    model->weights.attn_q_w = new llaisysTensor_t[nlayer]{};
    model->weights.attn_q_b = new llaisysTensor_t[nlayer]{};
    model->weights.attn_k_w = new llaisysTensor_t[nlayer]{};
    model->weights.attn_k_b = new llaisysTensor_t[nlayer]{};
    model->weights.attn_v_w = new llaisysTensor_t[nlayer]{};
    model->weights.attn_v_b = new llaisysTensor_t[nlayer]{};
    model->weights.attn_o_w = new llaisysTensor_t[nlayer]{};
    model->weights.mlp_norm_w = new llaisysTensor_t[nlayer]{};
    model->weights.mlp_gate_w = new llaisysTensor_t[nlayer]{};
    model->weights.mlp_up_w = new llaisysTensor_t[nlayer]{};
    model->weights.mlp_down_w = new llaisysTensor_t[nlayer]{};

    model->kv_cache_k.resize(nlayer);
    model->kv_cache_v.resize(nlayer);
    return model;
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return;
    }

    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    LlaisysQwen2Model *model) {
    if (model == nullptr) {
        throw std::invalid_argument("Qwen2 model must not be null");
    }
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(
    LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {
    if (model == nullptr || token_ids == nullptr || ntoken == 0) {
        throw std::invalid_argument("Qwen2 infer requires non-empty token ids");
    }
    ensure_weights(model);
    ensure_kv_cache(model, model->total_kv_len + ntoken);

    const size_t seq = ntoken;
    const size_t hs = model->meta.hs;
    const size_t nh = model->meta.nh;
    const size_t nkvh = model->meta.nkvh;
    const size_t dh = model->meta.dh;
    const size_t di = model->meta.di;

    auto input_ids = make_tensor(
        {seq}, LLAISYS_DTYPE_I64, model->device, model->device_id);
    input_ids->load(token_ids);

    std::vector<int64_t> positions(seq);
    for (size_t i = 0; i < seq; i++) {
        positions[i] = static_cast<int64_t>(model->total_kv_len + i);
    }
    auto pos_ids = make_tensor(
        {seq}, LLAISYS_DTYPE_I64, model->device, model->device_id);
    pos_ids->load(positions.data());

    auto x = make_tensor(
        {seq, hs}, model->meta.dtype, model->device, model->device_id);
    llaisys::ops::embedding(
        x, input_ids, unwrap(model->weights.in_embed, "in_embed"));

    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    for (size_t l = 0; l < model->meta.nlayer; l++) {
        auto residual_attn = make_tensor(
            {seq, hs}, model->meta.dtype, model->device, model->device_id);
        copy_tensor(residual_attn, x);

        auto normed = make_tensor(
            {seq, hs}, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::rms_norm(
            normed, x, unwrap(model->weights.attn_norm_w[l], "attn_norm_w"),
            model->meta.epsilon);

        auto q_flat = make_tensor(
            {seq, nh * dh}, model->meta.dtype, model->device, model->device_id);
        auto k_flat = make_tensor(
            {seq, nkvh * dh}, model->meta.dtype, model->device, model->device_id);
        auto v_flat = make_tensor(
            {seq, nkvh * dh}, model->meta.dtype, model->device, model->device_id);

        llaisys::ops::linear(
            q_flat, normed, unwrap(model->weights.attn_q_w[l], "attn_q_w"),
            model->weights.attn_q_b[l] == nullptr
                ? nullptr
                : model->weights.attn_q_b[l]->tensor);
        llaisys::ops::linear(
            k_flat, normed, unwrap(model->weights.attn_k_w[l], "attn_k_w"),
            model->weights.attn_k_b[l] == nullptr
                ? nullptr
                : model->weights.attn_k_b[l]->tensor);
        llaisys::ops::linear(
            v_flat, normed, unwrap(model->weights.attn_v_w[l], "attn_v_w"),
            model->weights.attn_v_b[l] == nullptr
                ? nullptr
                : model->weights.attn_v_b[l]->tensor);

        auto q = q_flat->view({seq, nh, dh});
        auto k = k_flat->view({seq, nkvh, dh});
        auto v = v_flat->view({seq, nkvh, dh});

        llaisys::ops::rope(q, q, pos_ids, model->meta.theta);
        llaisys::ops::rope(k, k, pos_ids, model->meta.theta);

        auto cache_k_write =
            model->kv_cache_k[l]->slice(
                0, model->total_kv_len, model->total_kv_len + seq);
        auto cache_v_write =
            model->kv_cache_v[l]->slice(
                0, model->total_kv_len, model->total_kv_len + seq);
        copy_tensor(cache_k_write, k);
        copy_tensor(cache_v_write, v);

        auto cache_k =
            model->kv_cache_k[l]->slice(
                0, 0, model->total_kv_len + seq);
        auto cache_v =
            model->kv_cache_v[l]->slice(
                0, 0, model->total_kv_len + seq);

        auto attn_val = make_tensor(
            {seq, nh, dh}, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::self_attention(
            attn_val, q, cache_k, cache_v, scale);

        auto attn_flat = attn_val->view({seq, hs});
        auto attn_proj = make_tensor(
            {seq, hs}, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::linear(
            attn_proj, attn_flat,
            unwrap(model->weights.attn_o_w[l], "attn_o_w"), nullptr);
        x = add_residual(
            attn_proj, residual_attn, model->device, model->device_id);

        auto residual_mlp = make_tensor(
            {seq, hs}, model->meta.dtype, model->device, model->device_id);
        copy_tensor(residual_mlp, x);

        auto mlp_normed = make_tensor(
            {seq, hs}, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::rms_norm(
            mlp_normed, x, unwrap(model->weights.mlp_norm_w[l], "mlp_norm_w"),
            model->meta.epsilon);

        auto gate = make_tensor(
            {seq, di}, model->meta.dtype, model->device, model->device_id);
        auto up = make_tensor(
            {seq, di}, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::linear(
            gate, mlp_normed,
            unwrap(model->weights.mlp_gate_w[l], "mlp_gate_w"), nullptr);
        llaisys::ops::linear(
            up, mlp_normed,
            unwrap(model->weights.mlp_up_w[l], "mlp_up_w"), nullptr);

        auto mlp = make_tensor(
            {seq, di}, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::swiglu(mlp, gate, up);

        auto down = make_tensor(
            {seq, hs}, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::linear(
            down, mlp,
            unwrap(model->weights.mlp_down_w[l], "mlp_down_w"), nullptr);
        x = add_residual(
            down, residual_mlp, model->device, model->device_id);
    }

    auto last = x->slice(0, seq - 1, seq);
    auto final_norm = make_tensor(
        {1, hs}, model->meta.dtype, model->device, model->device_id);
    llaisys::ops::rms_norm(
        final_norm, last, unwrap(model->weights.out_norm_w, "out_norm_w"),
        model->meta.epsilon);

    auto logits = make_tensor(
        {1, model->meta.voc}, model->meta.dtype,
        model->device, model->device_id);
    llaisys::ops::linear(
        logits, final_norm, unwrap(model->weights.out_embed, "out_embed"),
        nullptr);

    auto max_idx = make_tensor(
        {1}, LLAISYS_DTYPE_I64, model->device, model->device_id);
    auto max_val = make_tensor(
        {1}, model->meta.dtype, model->device, model->device_id);
    llaisys::ops::argmax(max_idx, max_val, logits);

    const int64_t next_token = read_i64(max_idx);
    model->total_kv_len += seq;
    return next_token;
}

} // extern "C"
