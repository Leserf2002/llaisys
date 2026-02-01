#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstring>

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 参数验证
    CHECK_SAME_DEVICE(out, in, weight);
    
    // 所有张量必须是连续的
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous");
    
    // 维度验证
    ASSERT(out->ndim() == 2, "RMSNorm: output must be 2-D tensor");
    ASSERT(in->ndim() == 2, "RMSNorm: input must be 2-D tensor");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1-D tensor");
    
    // 获取维度
    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];
    size_t weight_size = weight->shape()[0];
    
    // 检查维度匹配
    ASSERT(out->shape()[0] == batch_size && out->shape()[1] == hidden_size,
           "RMSNorm: output shape must match input shape");
    ASSERT(weight_size == hidden_size,
           "RMSNorm: weight size must match hidden size");
    
    // 数据类型验证
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    
    // 只支持CPU设备
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    
    // 获取数据指针
    std::byte* out_ptr = out->data();
    const std::byte* in_ptr = in->data();
    const std::byte* weight_ptr = weight->data();
    
    llaisysDataType_t dtype = out->dtype();
    
    // 根据数据类型实现RMS Norm
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float* out_f32 = reinterpret_cast<float*>(out_ptr);
            const float* in_f32 = reinterpret_cast<const float*>(in_ptr);
            const float* weight_f32 = reinterpret_cast<const float*>(weight_ptr);
            
            // 对每一行计算
            for (size_t b = 0; b < batch_size; ++b) {
                const float* in_row = in_f32 + b * hidden_size;
                float* out_row = out_f32 + b * hidden_size;
                
                // 计算平方和
                float sum_sq = 0.0f;
                for (size_t i = 0; i < hidden_size; ++i) {
                    float val = in_row[i];
                    sum_sq += val * val;
                }
                
                // 计算RMS：sqrt(mean(x^2) + eps)
                float rms = sqrtf(sum_sq / hidden_size + eps);
                float scale = 1.0f / rms;
                
                // 应用权重和归一化：y_i = (w_i * x_i) / rms
                for (size_t i = 0; i < hidden_size; ++i) {
                    out_row[i] = weight_f32[i] * in_row[i] * scale;
                }
            }
            break;
        }
            
        case LLAISYS_DTYPE_F16: {
            fp16_t* out_f16 = reinterpret_cast<fp16_t*>(out_ptr);
            const fp16_t* in_f16 = reinterpret_cast<const fp16_t*>(in_ptr);
            const fp16_t* weight_f16 = reinterpret_cast<const fp16_t*>(weight_ptr);
            
            // 使用float进行计算以提高精度
            for (size_t b = 0; b < batch_size; ++b) {
                const fp16_t* in_row_half = in_f16 + b * hidden_size;
                fp16_t* out_row_half = out_f16 + b * hidden_size;
                
                // 转换为float计算RMS
                float sum_sq = 0.0f;
                for (size_t i = 0; i < hidden_size; ++i) {
                    float val = llaisys::utils::cast<float>(in_row_half[i]);
                    sum_sq += val * val;
                }
                
                // 计算RMS
                float rms = sqrtf(sum_sq / hidden_size + eps);
                float scale = 1.0f / rms;
                
                // 应用权重和归一化
                for (size_t i = 0; i < hidden_size; ++i) {
                    float in_val = llaisys::utils::cast<float>(in_row_half[i]);
                    float weight_val = llaisys::utils::cast<float>(weight_f16[i]);
                    float out_val = weight_val * in_val * scale;
                    out_row_half[i] = llaisys::utils::cast<fp16_t>(out_val);
                }
            }
            break;
        }
            
        case LLAISYS_DTYPE_BF16: {
            bf16_t* out_bf16 = reinterpret_cast<bf16_t*>(out_ptr);
            const bf16_t* in_bf16 = reinterpret_cast<const bf16_t*>(in_ptr);
            const bf16_t* weight_bf16 = reinterpret_cast<const bf16_t*>(weight_ptr);
            
            // 使用float进行计算以提高精度
            for (size_t b = 0; b < batch_size; ++b) {
                const bf16_t* in_row_half = in_bf16 + b * hidden_size;
                bf16_t* out_row_half = out_bf16 + b * hidden_size;
                
                // 转换为float计算RMS
                float sum_sq = 0.0f;
                for (size_t i = 0; i < hidden_size; ++i) {
                    float val = llaisys::utils::cast<float>(in_row_half[i]);
                    sum_sq += val * val;
                }
                
                // 计算RMS
                float rms = sqrtf(sum_sq / hidden_size + eps);
                float scale = 1.0f / rms;
                
                // 应用权重和归一化
                for (size_t i = 0; i < hidden_size; ++i) {
                    float in_val = llaisys::utils::cast<float>(in_row_half[i]);
                    float weight_val = llaisys::utils::cast<float>(weight_bf16[i]);
                    float out_val = weight_val * in_val * scale;
                    out_row_half[i] = llaisys::utils::cast<bf16_t>(out_val);
                }
            }
            break;
        }
            
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops