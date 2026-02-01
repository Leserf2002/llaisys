#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    ptrdiff_t expected_stride = 1;
    for (int i = _meta.shape.size() - 1; i >= 0; --i) {
        if (_meta.strides[i] != expected_stride) return false;
        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[i]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = _meta.shape.size();
    if (order.size() != ndim_) throw std::runtime_error("Invalid permutation order");

    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);

    for (size_t i = 0; i < ndim_; ++i) {
        if (order[i] >= ndim_) throw std::runtime_error("Permutation index out of range");
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return tensor_t(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t total = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (total != numel()) throw std::runtime_error("view: total elements mismatch");

    std::vector<ptrdiff_t> new_strides(shape.size());
    if (isContiguous()) {
        ptrdiff_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            new_strides[i] = stride;
            stride *= static_cast<ptrdiff_t>(shape[i]);
        }
    } else {
        throw std::runtime_error("view on non-contiguous tensor not supported");
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return tensor_t(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= _meta.shape.size()) throw std::runtime_error("slice: dim out of range");
    if (start > end || end > _meta.shape[dim]) throw std::runtime_error("slice: invalid range");

    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    
    // 注意：strides是以元素为单位的，offset是以字节为单位的
    // 需要将元素偏移转换为字节偏移
    size_t element_offset = start * _meta.strides[dim];
    size_t byte_offset = _offset + element_offset * this->elementSize();
    
    return tensor_t(new Tensor(new_meta, _storage, byte_offset));
}

void Tensor::load(const void *src_) {
    size_t total_bytes = this->numel() * this->elementSize();
    
    // 根据设备类型选择复制方式
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU张量：直接内存拷贝
        const std::byte *src_ptr = reinterpret_cast<const std::byte *>(src_);
        std::memcpy(this->data(), src_ptr, total_bytes);
    } else {
        // 设备张量：使用运行时API进行异步拷贝
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(),          // 目标（设备内存）
            src_,                  // 源（主机内存）
            total_bytes,          // 字节数
            LLAISYS_MEMCPY_H2D    // 拷贝方向：主机到设备
        );
    }
}



tensor_t Tensor::contiguous() const {
    if (isContiguous()) return tensor_t(new Tensor(_meta, _storage, _offset));

    tensor_t out = create(shape(), dtype(), deviceType(), deviceId());
    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::byte *dst_ptr = out->data();

        for (size_t i = 0; i < numel(); ++i) {
            size_t idx = i;
            size_t offset = 0;
            for (size_t d = 0; d < ndim(); ++d) {
                size_t s = idx / out->strides()[d];
                idx %= out->strides()[d];
                offset += s * _meta.strides[d];
            }
            std::memcpy(dst_ptr + i * elementSize(), _storage->memory() + _offset + offset * elementSize(), elementSize());
        }
    } else {
        throw std::runtime_error("contiguous for non-CPU device not implemented");
    }

    return out;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    return view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device_type == deviceType() && (device == -1 || device == deviceId())) {
        return tensor_t(new Tensor(_meta, _storage, _offset));
    }

    tensor_t out = Tensor::create(shape(), dtype(), device_type, device);

    if (deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(out->data(), data(), numel() * elementSize());
    } else {
        core::context().runtime().api()->memcpy_sync(
            out->data(),
            data(),
            numel() * elementSize(),
            deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2D :
            (device_type == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_D2H : LLAISYS_MEMCPY_D2D));
    }
    return out;
}




} // namespace llaisys
