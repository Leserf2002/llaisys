#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cstring>

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 参数验证
    CHECK_SAME_DEVICE(out, index, weight);
    
    // index必须是1-D，int64类型
    ASSERT(index->ndim() == 1, "Embedding: index must be 1-D tensor");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64 type");
    
    // weight必须是2-D [vocab_size, embed_dim]
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2-D tensor [vocab_size, embed_dim]");
    
    // out必须是2-D [batch_size, embed_dim]
    ASSERT(out->ndim() == 2, "Embedding: output must be 2-D tensor [batch_size, embed_dim]");
    
    // 检查维度匹配
    size_t batch_size = out->shape()[0];
    size_t embed_dim = out->shape()[1];
    size_t vocab_size = weight->shape()[0];
    size_t weight_embed_dim = weight->shape()[1];
    
    ASSERT(batch_size == index->numel(), 
           "Embedding: output batch size must match index length");
    ASSERT(embed_dim == weight_embed_dim,
           "Embedding: output embed_dim must match weight embed_dim");
    
    // 数据类型必须匹配
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    
    // 暂时只支持连续张量
    ASSERT(out->isContiguous() && weight->isContiguous(),
           "Embedding: tensors must be contiguous for now");

    // 只支持CPU设备
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // 获取数据指针
    std::byte* out_ptr = out->data();
    const int64_t* index_ptr = reinterpret_cast<const int64_t*>(index->data());
    const std::byte* weight_ptr = weight->data();
    
    llaisysDataType_t dtype = out->dtype();
    size_t n = index->numel();
    
    // 根据数据类型实现
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float* out_f32 = reinterpret_cast<float*>(out_ptr);
            const float* weight_f32 = reinterpret_cast<const float*>(weight_ptr);
            
            for (size_t b = 0; b < n; ++b) {
                int64_t idx = index_ptr[b];
                if (idx < 0 || idx >= static_cast<int64_t>(vocab_size)) {
                    // 越界索引：填充0
                    float* dst = out_f32 + b * embed_dim;
                    for (size_t d = 0; d < embed_dim; ++d) {
                        dst[d] = 0.0f;
                    }
                } else {
                    // 复制embedding向量
                    const float* src = weight_f32 + idx * embed_dim;
                    float* dst = out_f32 + b * embed_dim;
                    for (size_t d = 0; d < embed_dim; ++d) {
                        dst[d] = src[d];
                    }
                }
            }
            break;
        }
            
        case LLAISYS_DTYPE_F16: {
            fp16_t* out_f16 = reinterpret_cast<fp16_t*>(out_ptr);
            const fp16_t* weight_f16 = reinterpret_cast<const fp16_t*>(weight_ptr);
            
            // 创建fp16的0值
            fp16_t zero_f16;
            zero_f16._v = 0;
            
            for (size_t b = 0; b < n; ++b) {
                int64_t idx = index_ptr[b];
                if (idx < 0 || idx >= static_cast<int64_t>(vocab_size)) {
                    // 越界索引：填充0
                    fp16_t* dst = out_f16 + b * embed_dim;
                    for (size_t d = 0; d < embed_dim; ++d) {
                        dst[d] = zero_f16;
                    }
                } else {
                    // 复制embedding向量
                    const fp16_t* src = weight_f16 + idx * embed_dim;
                    fp16_t* dst = out_f16 + b * embed_dim;
                    for (size_t d = 0; d < embed_dim; ++d) {
                        dst[d] = src[d];
                    }
                }
            }
            break;
        }
            
        case LLAISYS_DTYPE_BF16: {
            bf16_t* out_bf16 = reinterpret_cast<bf16_t*>(out_ptr);
            const bf16_t* weight_bf16 = reinterpret_cast<const bf16_t*>(weight_ptr);
            
            // 创建bf16的0值
            bf16_t zero_bf16;
            zero_bf16._v = 0;
            
            for (size_t b = 0; b < n; ++b) {
                int64_t idx = index_ptr[b];
                if (idx < 0 || idx >= static_cast<int64_t>(vocab_size)) {
                    // 越界索引：填充0
                    bf16_t* dst = out_bf16 + b * embed_dim;
                    for (size_t d = 0; d < embed_dim; ++d) {
                        dst[d] = zero_bf16;
                    }
                } else {
                    // 复制embedding向量
                    const bf16_t* src = weight_bf16 + idx * embed_dim;
                    bf16_t* dst = out_bf16 + b * embed_dim;
                    for (size_t d = 0; d < embed_dim; ++d) {
                        dst[d] = src[d];
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