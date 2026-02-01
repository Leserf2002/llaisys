#include "op.hpp"
#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"
#include <cstring>
#include <limits>

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 基础验证
    if (!vals || vals->numel() == 0) {
        throw std::runtime_error("argmax: invalid input tensor");
    }
    
    size_t n = vals->numel();
    llaisysDataType_t dtype = vals->dtype();
    
    // 使用utils中的cast函数进行类型转换和比较
    using llaisys::utils::cast;
    
    if (dtype == LLAISYS_DTYPE_F32) {
        const float* data = reinterpret_cast<const float*>(vals->data());
        float max_value = data[0];
        size_t max_index = 0;
        
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max_value) {
                max_value = data[i];
                max_index = i;
            }
        }
        
        // 存储结果
        float* max_val_ptr = reinterpret_cast<float*>(max_val->data());
        *max_val_ptr = max_value;
        
        int64_t* max_idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
        *max_idx_ptr = static_cast<int64_t>(max_index);
        
    } else if (dtype == LLAISYS_DTYPE_F16) {
        const fp16_t* data = reinterpret_cast<const fp16_t*>(vals->data());
        
        // 转换为float进行比较
        float max_value = cast<float>(data[0]);
        size_t max_index = 0;
        
        for (size_t i = 1; i < n; ++i) {
            float current = cast<float>(data[i]);
            if (current > max_value) {
                max_value = current;
                max_index = i;
            }
        }
        
        // 存储结果（转回fp16_t）
        fp16_t* max_val_ptr = reinterpret_cast<fp16_t*>(max_val->data());
        *max_val_ptr = cast<fp16_t>(max_value);
        
        int64_t* max_idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
        *max_idx_ptr = static_cast<int64_t>(max_index);
        
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        const bf16_t* data = reinterpret_cast<const bf16_t*>(vals->data());
        
        // 转换为float进行比较
        float max_value = cast<float>(data[0]);
        size_t max_index = 0;
        
        for (size_t i = 1; i < n; ++i) {
            float current = cast<float>(data[i]);
            if (current > max_value) {
                max_value = current;
                max_index = i;
            }
        }
        
        // 存储结果（转回bf16_t）
        bf16_t* max_val_ptr = reinterpret_cast<bf16_t*>(max_val->data());
        *max_val_ptr = cast<bf16_t>(max_value);
        
        int64_t* max_idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
        *max_idx_ptr = static_cast<int64_t>(max_index);
        
    } else if (dtype == LLAISYS_DTYPE_I32) {
        const int32_t* data = reinterpret_cast<const int32_t*>(vals->data());
        int32_t max_value = data[0];
        size_t max_index = 0;
        
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max_value) {
                max_value = data[i];
                max_index = i;
            }
        }
        
        int32_t* max_val_ptr = reinterpret_cast<int32_t*>(max_val->data());
        *max_val_ptr = max_value;
        
        int64_t* max_idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
        *max_idx_ptr = static_cast<int64_t>(max_index);
        
    } else if (dtype == LLAISYS_DTYPE_I64) {
        const int64_t* data = reinterpret_cast<const int64_t*>(vals->data());
        int64_t max_value = data[0];
        size_t max_index = 0;
        
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max_value) {
                max_value = data[i];
                max_index = i;
            }
        }
        
        int64_t* max_val_ptr = reinterpret_cast<int64_t*>(max_val->data());
        *max_val_ptr = max_value;
        
        int64_t* max_idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
        *max_idx_ptr = static_cast<int64_t>(max_index);
        
    } else {
        // 其他类型可以类似实现
        throw std::runtime_error("argmax: unsupported data type");
    }
}
} // namespace llaisys::ops
