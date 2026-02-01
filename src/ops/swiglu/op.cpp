#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <limits>

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {

    CHECK_SAME_DEVICE(out, gate, up);

    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: tensors must be contiguous");

    ASSERT(out->ndim() == 2 &&
           gate->ndim() == 2 &&
           up->ndim() == 2,
           "SwiGLU: tensors must be 2-D");

    ASSERT(out->shape() == gate->shape() &&
           out->shape() == up->shape(),
           "SwiGLU: shape mismatch");

    CHECK_SAME_DTYPE(out->dtype(), gate->dtype());
    CHECK_SAME_DTYPE(out->dtype(), up->dtype());

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    size_t numel = out->numel();

    std::byte* out_ptr = out->data();
    const std::byte* gate_ptr = gate->data();
    const std::byte* up_ptr = up->data();

    auto dtype = out->dtype();

    switch (dtype) {

    // ========================= F32 =========================
    case LLAISYS_DTYPE_F32: {

        float* out_f = reinterpret_cast<float*>(out_ptr);
        const float* gate_f = reinterpret_cast<const float*>(gate_ptr);
        const float* up_f = reinterpret_cast<const float*>(up_ptr);

        for (size_t i = 0; i < numel; ++i) {

            float g = gate_f[i];
            float swish = g / (1.0f + std::exp(-g));

            out_f[i] = up_f[i] * swish;
        }

        break;
    }

    // ========================= F16 =========================
    case LLAISYS_DTYPE_F16: {

        fp16_t* out_h = reinterpret_cast<fp16_t*>(out_ptr);
        const fp16_t* gate_h = reinterpret_cast<const fp16_t*>(gate_ptr);
        const fp16_t* up_h = reinterpret_cast<const fp16_t*>(up_ptr);

        for (size_t i = 0; i < numel; ++i) {

            float g = llaisys::utils::cast<float>(gate_h[i]);
            float u = llaisys::utils::cast<float>(up_h[i]);

            float swish = g / (1.0f + std::exp(-g));
            float result = u * swish;

            out_h[i] = llaisys::utils::cast<fp16_t>(result);
        }

        break;
    }

    // ========================= BF16 =========================
    case LLAISYS_DTYPE_BF16: {

        bf16_t* out_b = reinterpret_cast<bf16_t*>(out_ptr);
        const bf16_t* gate_b = reinterpret_cast<const bf16_t*>(gate_ptr);
        const bf16_t* up_b = reinterpret_cast<const bf16_t*>(up_ptr);

        for (size_t i = 0; i < numel; ++i) {

            float g = llaisys::utils::cast<float>(gate_b[i]);
            float u = llaisys::utils::cast<float>(up_b[i]);

            float swish = g / (1.0f + std::exp(-g));
            float result = u * swish;

            out_b[i] = llaisys::utils::cast<bf16_t>(result);
        }

        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops
