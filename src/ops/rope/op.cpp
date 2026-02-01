#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstring>

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 参数验证
    CHECK_SAME_DEVICE(out, in, pos_ids);
    
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous");
    
    ASSERT(out->ndim() == 3, "RoPE: output must be 3-D tensor [seqlen, nhead, d]");
    ASSERT(in->ndim() == 3, "RoPE: input must be 3-D tensor [seqlen, nhead, d]");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1-D tensor [seqlen]");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64 type");
    
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t d = in->shape()[2];
    
    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == n_heads && out->shape()[2] == d,
           "RoPE: output shape must match input shape");
    ASSERT(pos_ids->numel() == seq_len,
           "RoPE: pos_ids length must match sequence length");
    
    ASSERT(d % 2 == 0, "RoPE: dimension d must be even");
    size_t d_half = d / 2;
    
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    
    std::byte* out_ptr = out->data();
    const std::byte* in_ptr = in->data();
    const int64_t* pos_ids_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());
    
    llaisysDataType_t dtype = out->dtype();
    
    // 预计算频率指数：theta^(2i/d)
    std::vector<float> theta_pow(d_half);
    for (size_t i = 0; i < d_half; ++i) {
        theta_pow[i] = powf(theta, 2.0f * i / d);
    }
    
    // 根据数据类型实现RoPE
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float* out_f32 = reinterpret_cast<float*>(out_ptr);
            const float* in_f32 = reinterpret_cast<const float*>(in_ptr);
            
            // 为每个位置预计算sin和cos
            std::vector<float> sin_vals(seq_len * d_half);
            std::vector<float> cos_vals(seq_len * d_half);
            
            for (size_t pos = 0; pos < seq_len; ++pos) {
                float position = static_cast<float>(pos_ids_ptr[pos]);
                float* pos_sin = &sin_vals[pos * d_half];
                float* pos_cos = &cos_vals[pos * d_half];
                
                for (size_t i = 0; i < d_half; ++i) {
                    float freq = position / theta_pow[i];
                    pos_sin[i] = sinf(freq);
                    pos_cos[i] = cosf(freq);
                }
            }
            
            // 对每个位置和每个头应用旋转
            for (size_t pos = 0; pos < seq_len; ++pos) {
                const float* pos_sin = &sin_vals[pos * d_half];
                const float* pos_cos = &cos_vals[pos * d_half];
                
                for (size_t h = 0; h < n_heads; ++h) {
                    const float* in_head = in_f32 + (pos * n_heads + h) * d;
                    float* out_head = out_f32 + (pos * n_heads + h) * d;
                    
                    const float* x_a = in_head;          // 前半部分 [0, d_half-1]
                    const float* x_b = in_head + d_half; // 后半部分 [d_half, d-1]
                    
                    float* y_a = out_head;
                    float* y_b = out_head + d_half;
                    
                    for (size_t i = 0; i < d_half; ++i) {
                        // y_a[i] = x_a[i] * cos - x_b[i] * sin
                        // y_b[i] = x_b[i] * cos + x_a[i] * sin
                        y_a[i] = x_a[i] * pos_cos[i] - x_b[i] * pos_sin[i];
                        y_b[i] = x_b[i] * pos_cos[i] + x_a[i] * pos_sin[i];
                    }
                }
            }
            break;
        }
            
        case LLAISYS_DTYPE_F16: {
            fp16_t* out_f16 = reinterpret_cast<fp16_t*>(out_ptr);
            const fp16_t* in_f16 = reinterpret_cast<const fp16_t*>(in_ptr);
            
            // 预计算sin和cos（使用float）
            std::vector<float> sin_vals(seq_len * d_half);
            std::vector<float> cos_vals(seq_len * d_half);
            
            for (size_t pos = 0; pos < seq_len; ++pos) {
                float position = static_cast<float>(pos_ids_ptr[pos]);
                float* pos_sin = &sin_vals[pos * d_half];
                float* pos_cos = &cos_vals[pos * d_half];
                
                for (size_t i = 0; i < d_half; ++i) {
                    float freq = position / theta_pow[i];
                    pos_sin[i] = sinf(freq);
                    pos_cos[i] = cosf(freq);
                }
            }
            
            // 对每个位置和每个头应用旋转
            for (size_t pos = 0; pos < seq_len; ++pos) {
                const float* pos_sin = &sin_vals[pos * d_half];
                const float* pos_cos = &cos_vals[pos * d_half];
                
                for (size_t h = 0; h < n_heads; ++h) {
                    const fp16_t* in_head = in_f16 + (pos * n_heads + h) * d;
                    fp16_t* out_head = out_f16 + (pos * n_heads + h) * d;
                    
                    // 应用旋转
                    for (size_t i = 0; i < d_half; ++i) {
                        // 转换为float计算
                        float x_a = llaisys::utils::cast<float>(in_head[i]);
                        float x_b = llaisys::utils::cast<float>(in_head[d_half + i]);
                        
                        // 旋转
                        float y_a = x_a * pos_cos[i] - x_b * pos_sin[i];
                        float y_b = x_b * pos_cos[i] + x_a * pos_sin[i];
                        
                        // 转换回fp16
                        out_head[i] = llaisys::utils::cast<fp16_t>(y_a);
                        out_head[d_half + i] = llaisys::utils::cast<fp16_t>(y_b);
                    }
                }
            }
            break;
        }
            
        case LLAISYS_DTYPE_BF16: {
            bf16_t* out_bf16 = reinterpret_cast<bf16_t*>(out_ptr);
            const bf16_t* in_bf16 = reinterpret_cast<const bf16_t*>(in_ptr);
            
            // 预计算sin和cos（使用float）
            std::vector<float> sin_vals(seq_len * d_half);
            std::vector<float> cos_vals(seq_len * d_half);
            
            for (size_t pos = 0; pos < seq_len; ++pos) {
                float position = static_cast<float>(pos_ids_ptr[pos]);
                float* pos_sin = &sin_vals[pos * d_half];
                float* pos_cos = &cos_vals[pos * d_half];
                
                for (size_t i = 0; i < d_half; ++i) {
                    float freq = position / theta_pow[i];
                    pos_sin[i] = sinf(freq);
                    pos_cos[i] = cosf(freq);
                }
            }
            
            // 对每个位置和每个头应用旋转
            for (size_t pos = 0; pos < seq_len; ++pos) {
                const float* pos_sin = &sin_vals[pos * d_half];
                const float* pos_cos = &cos_vals[pos * d_half];
                
                for (size_t h = 0; h < n_heads; ++h) {
                    const bf16_t* in_head = in_bf16 + (pos * n_heads + h) * d;
                    bf16_t* out_head = out_bf16 + (pos * n_heads + h) * d;
                    
                    // 应用旋转
                    for (size_t i = 0; i < d_half; ++i) {
                        // 转换为float计算
                        float x_a = llaisys::utils::cast<float>(in_head[i]);
                        float x_b = llaisys::utils::cast<float>(in_head[d_half + i]);
                        
                        // 旋转
                        float y_a = x_a * pos_cos[i] - x_b * pos_sin[i];
                        float y_b = x_b * pos_cos[i] + x_a * pos_sin[i];
                        
                        // 转换回bf16
                        out_head[i] = llaisys::utils::cast<bf16_t>(y_a);
                        out_head[d_half + i] = llaisys::utils::cast<bf16_t>(y_b);
                    }
                }
            }
            break;
        }
            
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops