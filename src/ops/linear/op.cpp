#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cstring>

namespace llaisys::ops {

// 优化的内积计算（循环展开）
template<typename T>
float dot_product_optimized(const T* a, const T* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    
    // 展开4次循环
    for (; i + 3 < n; i += 4) {
        sum += llaisys::utils::cast<float>(a[i]) * llaisys::utils::cast<float>(b[i]) +
               llaisys::utils::cast<float>(a[i+1]) * llaisys::utils::cast<float>(b[i+1]) +
               llaisys::utils::cast<float>(a[i+2]) * llaisys::utils::cast<float>(b[i+2]) +
               llaisys::utils::cast<float>(a[i+3]) * llaisys::utils::cast<float>(b[i+3]);
    }
    
    // 处理剩余元素
    for (; i < n; ++i) {
        sum += llaisys::utils::cast<float>(a[i]) * llaisys::utils::cast<float>(b[i]);
    }
    
    return sum;
}

// 针对float32的特殊优化（避免类型转换）
template<>
float dot_product_optimized<float>(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    
    // 展开4次循环
    for (; i + 3 < n; i += 4) {
        sum += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
    }
    
    // 处理剩余元素
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 参数验证
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: all tensors must be contiguous");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous");
    }
    
    ASSERT(out->ndim() == 2, "Linear: output must be 2-D tensor");
    ASSERT(in->ndim() == 2, "Linear: input must be 2-D tensor");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2-D tensor");
    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1-D tensor");
    }
    
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = out->shape()[1];
    size_t weight_in_features = weight->shape()[1];
    size_t weight_out_features = weight->shape()[0];
    
    ASSERT(out->shape()[0] == batch_size,
           "Linear: output batch size must match input batch size");
    ASSERT(in_features == weight_in_features,
           "Linear: input features must match weight input features");
    ASSERT(out_features == weight_out_features,
           "Linear: output features must match weight output features");
    if (bias) {
        ASSERT(bias->numel() == out_features,
               "Linear: bias size must match output features");
    }
    
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    
    std::byte* out_ptr = out->data();
    const std::byte* in_ptr = in->data();
    const std::byte* weight_ptr = weight->data();
    const std::byte* bias_ptr = bias ? bias->data() : nullptr;
    
    llaisysDataType_t dtype = out->dtype();
    
    // 优化的实现
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float* out_f32 = reinterpret_cast<float*>(out_ptr);
            const float* in_f32 = reinterpret_cast<const float*>(in_ptr);
            const float* weight_f32 = reinterpret_cast<const float*>(weight_ptr);
            const float* bias_f32 = bias_ptr ? reinterpret_cast<const float*>(bias_ptr) : nullptr;
            
            // 预加载偏置或清零
            if (bias_f32) {
                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t o = 0; o < out_features; ++o) {
                        out_f32[b * out_features + o] = bias_f32[o];
                    }
                }
            } else {
                memset(out_f32, 0, batch_size * out_features * sizeof(float));
            }
            
            // GEMM: Y += X * W^T
            for (size_t b = 0; b < batch_size; ++b) {
                const float* x_row = in_f32 + b * in_features;
                float* y_row = out_f32 + b * out_features;
                
                for (size_t o = 0; o < out_features; ++o) {
                    const float* w_row = weight_f32 + o * in_features;
                    y_row[o] += dot_product_optimized(x_row, w_row, in_features);
                }
            }
            break;
        }
            
        case LLAISYS_DTYPE_F16: {
            fp16_t* out_f16 = reinterpret_cast<fp16_t*>(out_ptr);
            const fp16_t* in_f16 = reinterpret_cast<const fp16_t*>(in_ptr);
            const fp16_t* weight_f16 = reinterpret_cast<const fp16_t*>(weight_ptr);
            const fp16_t* bias_f16 = bias_ptr ? reinterpret_cast<const fp16_t*>(bias_ptr) : nullptr;
            
            // 使用临时float缓冲区避免重复转换
            std::vector<float> out_buffer(batch_size * out_features, 0.0f);
            
            // 预加载偏置
            if (bias_f16) {
                for (size_t o = 0; o < out_features; ++o) {
                    float bias_val = llaisys::utils::cast<float>(bias_f16[o]);
                    for (size_t b = 0; b < batch_size; ++b) {
                        out_buffer[b * out_features + o] = bias_val;
                    }
                }
            }
            
            // GEMM计算
            for (size_t b = 0; b < batch_size; ++b) {
                const fp16_t* x_row = in_f16 + b * in_features;
                float* y_row = out_buffer.data() + b * out_features;
                
                for (size_t o = 0; o < out_features; ++o) {
                    const fp16_t* w_row = weight_f16 + o * in_features;
                    y_row[o] += dot_product_optimized(x_row, w_row, in_features);
                }
            }
            
            // 转换回fp16
            for (size_t i = 0; i < batch_size * out_features; ++i) {
                out_f16[i] = llaisys::utils::cast<fp16_t>(out_buffer[i]);
            }
            break;
        }
            
        case LLAISYS_DTYPE_BF16: {
            bf16_t* out_bf16 = reinterpret_cast<bf16_t*>(out_ptr);
            const bf16_t* in_bf16 = reinterpret_cast<const bf16_t*>(in_ptr);
            const bf16_t* weight_bf16 = reinterpret_cast<const bf16_t*>(weight_ptr);
            const bf16_t* bias_bf16 = bias_ptr ? reinterpret_cast<const bf16_t*>(bias_ptr) : nullptr;
            
            // 使用临时float缓冲区
            std::vector<float> out_buffer(batch_size * out_features, 0.0f);
            
            // 预加载偏置
            if (bias_bf16) {
                for (size_t o = 0; o < out_features; ++o) {
                    float bias_val = llaisys::utils::cast<float>(bias_bf16[o]);
                    for (size_t b = 0; b < batch_size; ++b) {
                        out_buffer[b * out_features + o] = bias_val;
                    }
                }
            }
            
            // GEMM计算
            for (size_t b = 0; b < batch_size; ++b) {
                const bf16_t* x_row = in_bf16 + b * in_features;
                float* y_row = out_buffer.data() + b * out_features;
                
                for (size_t o = 0; o < out_features; ++o) {
                    const bf16_t* w_row = weight_bf16 + o * in_features;
                    y_row[o] += dot_product_optimized(x_row, w_row, in_features);
                }
            }
            
            // 转换回bf16
            for (size_t i = 0; i < batch_size * out_features; ++i) {
                out_bf16[i] = llaisys::utils::cast<bf16_t>(out_buffer[i]);
            }
            break;
        }
            
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops