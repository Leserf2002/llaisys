#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstring>
#include <limits>

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous");

    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3-D tensor [seqlen, nhead, dv]");
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3-D tensor [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3-D tensor [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3-D tensor [total_len, nkvhead, dv]");

    size_t seq_len = q->shape()[0];
    size_t n_q_head = q->shape()[1];
    size_t d = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t n_kv_head = k->shape()[1];
    size_t d_k = k->shape()[2];

    size_t dv = v->shape()[2];

    ASSERT(attn_val->shape()[0] == seq_len &&
           attn_val->shape()[1] == n_q_head &&
           attn_val->shape()[2] == dv,
           "SelfAttention: attn_val shape mismatch");

    ASSERT(v->shape()[0] == total_len &&
           v->shape()[1] == n_kv_head,
           "SelfAttention: v shape mismatch with k");

    ASSERT(d == d_k, "SelfAttention: q and k must have same last dimension");
    ASSERT(n_q_head % n_kv_head == 0, "SelfAttention: n_q_head must be divisible by n_kv_head");

    size_t group_size = n_q_head / n_kv_head;
    size_t kv_offset = total_len - seq_len;

    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), k->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), v->dtype());

    if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    std::byte* attn_val_ptr = attn_val->data();
    const std::byte* q_ptr = q->data();
    const std::byte* k_ptr = k->data();
    const std::byte* v_ptr = v->data();

    auto dtype = attn_val->dtype();

    switch (dtype) {

    case LLAISYS_DTYPE_F32: {

        float* out = reinterpret_cast<float*>(attn_val_ptr);
        const float* q_f = reinterpret_cast<const float*>(q_ptr);
        const float* k_f = reinterpret_cast<const float*>(k_ptr);
        const float* v_f = reinterpret_cast<const float*>(v_ptr);

        for (size_t q_head = 0; q_head < n_q_head; ++q_head) {
            size_t kv_head = q_head / group_size;

            for (size_t q_pos = 0; q_pos < seq_len; ++q_pos) {

                size_t q_offset = (q_pos * n_q_head + q_head) * d;
                size_t out_offset = (q_pos * n_q_head + q_head) * dv;

                const float* q_vec = q_f + q_offset;
                float* out_vec = out + out_offset;

                size_t context_len = std::min(q_pos + kv_offset + 1, total_len);

                std::vector<float> scores(context_len);

                float max_score = -std::numeric_limits<float>::infinity();

                for (size_t k_pos = 0; k_pos < context_len; ++k_pos) {

                    const float* k_vec =
                        k_f + (k_pos * n_kv_head + kv_head) * d;

                    float s = 0.f;
                    for (size_t i = 0; i < d; ++i)
                        s += q_vec[i] * k_vec[i];

                    s *= scale;

                    scores[k_pos] = s;
                    if (s > max_score) max_score = s;
                }

                float sum_exp = 0.f;

                for (size_t i = 0; i < context_len; ++i) {
                    scores[i] = std::exp(scores[i] - max_score);
                    sum_exp += scores[i];
                }

                float inv_sum = sum_exp > 0 ? 1.f / sum_exp : 0.f;

                std::memset(out_vec, 0, dv * sizeof(float));

                for (size_t k_pos = 0; k_pos < context_len; ++k_pos) {

                    float w = scores[k_pos] * inv_sum;

                    const float* v_vec =
                        v_f + (k_pos * n_kv_head + kv_head) * dv;

                    for (size_t i = 0; i < dv; ++i)
                        out_vec[i] += w * v_vec[i];
                }
            }
        }

        break;
    }

    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16: {

        // 统一半精度实现（用 float 中间计算）

        bool is_fp16 = (dtype == LLAISYS_DTYPE_F16);

        for (size_t q_head = 0; q_head < n_q_head; ++q_head) {

            size_t kv_head = q_head / group_size;

            for (size_t q_pos = 0; q_pos < seq_len; ++q_pos) {

                size_t q_offset = (q_pos * n_q_head + q_head) * d;
                size_t out_offset = (q_pos * n_q_head + q_head) * dv;

                std::vector<float> q_vec(d);

                for (size_t i = 0; i < d; ++i) {
                    if (is_fp16)
                        q_vec[i] = llaisys::utils::cast<float>(
                            reinterpret_cast<const fp16_t*>(q_ptr)[q_offset + i]);
                    else
                        q_vec[i] = llaisys::utils::cast<float>(
                            reinterpret_cast<const bf16_t*>(q_ptr)[q_offset + i]);
                }

                size_t context_len = std::min(q_pos + kv_offset + 1, total_len);

                std::vector<float> scores(context_len);
                float max_score = -std::numeric_limits<float>::infinity();

                for (size_t k_pos = 0; k_pos < context_len; ++k_pos) {

                    float s = 0.f;

                    for (size_t i = 0; i < d; ++i) {

                        float kval;

                        if (is_fp16)
                            kval = llaisys::utils::cast<float>(
                                reinterpret_cast<const fp16_t*>(k_ptr)[
                                    (k_pos * n_kv_head + kv_head) * d + i]);
                        else
                            kval = llaisys::utils::cast<float>(
                                reinterpret_cast<const bf16_t*>(k_ptr)[
                                    (k_pos * n_kv_head + kv_head) * d + i]);

                        s += q_vec[i] * kval;
                    }

                    s *= scale;

                    scores[k_pos] = s;
                    if (s > max_score) max_score = s;
                }

                float sum_exp = 0.f;

                for (size_t i = 0; i < context_len; ++i) {
                    scores[i] = std::exp(scores[i] - max_score);
                    sum_exp += scores[i];
                }

                float inv_sum = sum_exp > 0 ? 1.f / sum_exp : 0.f;

                std::vector<float> out_vec(dv, 0.f);

                for (size_t k_pos = 0; k_pos < context_len; ++k_pos) {

                    float w = scores[k_pos] * inv_sum;

                    for (size_t i = 0; i < dv; ++i) {

                        float vval;

                        if (is_fp16)
                            vval = llaisys::utils::cast<float>(
                                reinterpret_cast<const fp16_t*>(v_ptr)[
                                    (k_pos * n_kv_head + kv_head) * dv + i]);
                        else
                            vval = llaisys::utils::cast<float>(
                                reinterpret_cast<const bf16_t*>(v_ptr)[
                                    (k_pos * n_kv_head + kv_head) * dv + i]);

                        out_vec[i] += w * vval;
                    }
                }

                for (size_t i = 0; i < dv; ++i) {

                    if (is_fp16)
                        reinterpret_cast<fp16_t*>(attn_val_ptr)[out_offset + i] =
                            llaisys::utils::cast<fp16_t>(out_vec[i]);
                    else
                        reinterpret_cast<bf16_t*>(attn_val_ptr)[out_offset + i] =
                            llaisys::utils::cast<bf16_t>(out_vec[i]);
                }
            }
        }

        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

}  // namespace llaisys::ops