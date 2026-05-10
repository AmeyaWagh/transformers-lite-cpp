#pragma once
#include <cassert>
#include <concepts>
#include <iostream>
#include <string>
#include <vector>

#include "memory.hpp"

namespace transformers_lite {

class Shape;

/** @brief Stream insertion operator for Shape (forward declaration). */
std::ostream &operator<<(std::ostream &os, const Shape &shape);

/**
 * @brief Stores the Shape information of a tensor.
 *
 */
class Shape {
 public:
    /** @brief Construct a scalar (0-dimensional) shape. */
    Shape() : m_shape(), m_stride(), m_num_dims(0), m_dim_names() {}

    /**
     * @brief Construct a shape from an initializer list of dimensions.
     *
     * @param shape initializer list of dimension sizes
     */
    Shape(std::initializer_list<size_t> shape) : m_shape(shape), m_stride(), m_num_dims(0), m_dim_names() { initialize(); }

    /**
     * @brief Construct a shape from a vector of dimensions.
     *
     * @param shape vector of dimension sizes
     */
    Shape(std::vector<size_t> shape) : m_shape(shape), m_stride(), m_num_dims(0) { initialize(); }

    /**
     * @brief Copy construct a Shape.
     *
     * @param shape shape to copy from
     */
    Shape(const Shape &shape) : m_shape(shape.shapeVec()), m_stride(shape.strideVec()), m_num_dims(shape.numDims()), m_dim_names() {}

    /**
     * @brief Construct a 1-dimensional shape.
     *
     * @param dim size of the single dimension
     */
    Shape(const size_t dim) : m_shape({dim}), m_stride(), m_num_dims(1), m_dim_names() { initialize(); }

    /**
     * @brief Construct a shape from variadic dimension arguments.
     *
     * @tparam ARGS dimension types (must be convertible to size_t)
     * @param args dimension sizes
     */
    template <std::integral... ARGS> Shape(ARGS... args) : m_shape({static_cast<size_t>(args)...}), m_stride(), m_num_dims(sizeof...(args)), m_dim_names() {
        initialize();
    }

    /**
     * @brief Get the memory offset from the tensor indices.
     *
     * @tparam ARGS
     * @param idx tensor index
     * @param args tensor indices
     * @return const size_t - offset of the element in the memory.
     */
    template <typename... ARGS> auto operator()(size_t idx, ARGS... args) const -> const size_t {
        assert(sizeof...(args) < m_shape.size());
        assert(idx < m_shape[(m_num_dims - 1 - sizeof...(ARGS))]);
        return idx * m_stride[(m_num_dims - 1 - sizeof...(ARGS))] + this->operator()(args...);
    }

    /**
     * @brief Get the memory offset from the tensor index.
     *
     * @param idx tensor index
     * @return const size_t - offset of the element in the memory.
     */
    auto operator()(size_t idx) const -> const size_t {
        if (isScalar()) {
            assert(idx == 0);
            return 0;
        }
        assert(idx < m_shape[m_num_dims - 1]);
        return idx * m_stride[m_num_dims - 1];
    }

    /**
     * @brief Get shape at given dimension index
     *
     * @param idx dimension index
     * @return const size_t - shape of the given dimension.
     */
    auto operator[](size_t idx) const -> const size_t { return m_shape.at(idx); }

    /**
     * @brief Get slice of multidimensional shape
     *
     * @tparam ARGS index types
     * @param idx index of the shape dimension.
     * @param args other indices of the shape dimension.
     * @return Shape sliced shape Shape[nDIM - 1 + sizeof...(args)]
     */
    template <typename... ARGS> auto slice(size_t idx, ARGS... args) const -> Shape {
        /**
         * Let Original shape of a tensor be Shape(2,3,5) stride = {15,5,1} nDIM=3 and names = (C,H,W)
         *
         * shape.slice(0)       -> Shape(3,5),  stride = {5,1}      nDIM=2 (H,W)
         * shape.slice(1)       -> Shape(3,5),  stride = {5,1}      nDIM=2 (H,W)
         * shape.slice(1,0)     -> Shape(5),    stride = {1}        nDIM=1 (W)
         * shape.slice(1,1)     -> Shape(5),    stride = {1}        nDIM=1 (W)
         * shape.slice(1,1,2)   -> Shape(None), stride = {1}        nDIM=0 (None)
         *
         */

        if constexpr (sizeof...(ARGS) == 0) {
            // Special case shape.slice(arg0) -> Shape[nDIM-1]
            std::vector<size_t> new_shape = this->shapeVec();
            new_shape.erase(new_shape.begin());
            return Shape(new_shape);
        } else {
            // General case shape.slice(arg0, arg1)         -> Shape[nDIM-2]
            // General case shape.slice(arg0, arg1, arg2)   -> Shape[nDIM-3]
            // General case shape.slice(arg0, arg...n)      -> Shape[nDIM-n]
            std::vector<size_t> new_shape = this->shapeVec();
            new_shape.erase(new_shape.begin());
            return Shape(new_shape).slice(args...);
        }
    }

    /**
     * @brief Offset is used to get the offset of the tensor memory pointer for different slices.
     *
     * @tparam ARGS index types
     * @param idx index of the shape dimension.
     * @param args other indices of the shape dimension.
     * @return size_t - offset value from the beginning of the original tensor.
     */
    template <typename... ARGS> auto offset(size_t idx, ARGS... args) const -> size_t {
        /**
         * Lets take the same example
         * Let Original shape of a tensor be Shape(2,3,5) stride = {15,5,1} nDIM=3
         *
         * tensor               -> Shape(2,3,5) stride = {15,5,1}   offset {0}                  = 0
         * shape.slice(0)       -> Shape(3,5),  stride = {5,1}      offset {0*15}               = 0
         * shape.slice(1)       -> Shape(3,5),  stride = {5,1}      offset {1*15}               = 15
         * shape.slice(1,0)     -> Shape(5),    stride = {1}        offset {1*15 + 0*5}         = 15
         * shape.slice(1,1)     -> Shape(5),    stride = {1}        offset {1*15 + 1*5}         = 20
         * shape.slice(1,1,2)   -> Shape(None), stride = {1}        offset {1*15 + 1*5 + 2*1}   = 22
         *
         */

        if constexpr (sizeof...(ARGS) == 0) {
            if (isScalar()) {
                return idx * 1;
            }
            return idx * m_stride[0];
        } else {
            std::vector<size_t> new_shape_ = this->shapeVec();
            new_shape_.erase(new_shape_.begin());
            Shape new_shape(new_shape_);
            return idx * m_stride[0] + new_shape.offset(args...);
        }
    }

    /**
     * @brief Total number of elements in the shape.
     *
     * @return size_t total element count
     */
    auto size() const -> const size_t {
        if (m_shape.empty()) {
            return 0;
        }
        return m_shape[0] * m_stride[0];
    }

    /**
     * @brief Number of dimensions.
     *
     * @return size_t dimension count
     */
    auto numDims() const -> const size_t { return m_num_dims; }

    /**
     * @brief Total number of elements (same as size, but returns 1 for scalars).
     *
     * @return size_t element count
     */
    auto numElements() const -> const size_t {
        if (isScalar()) {
            return 1UL;
        }
        // If we have a tensor of shape (C,H,W) stride (H*W,W,1),
        // then num elements i.e. C*H*W is simple shape[0]*stride[0]
        return m_shape[0] * m_stride[0];
    }

    /** @brief Get the dimension sizes as a vector. */
    auto shapeVec() const -> const std::vector<size_t> { return m_shape; }

    /** @brief Get the stride values as a vector. */
    auto strideVec() const -> const std::vector<size_t> { return m_stride; }

    /** @brief Get the dimension names as a vector. */
    auto getNamesVec() const -> const std::vector<std::string> { return m_dim_names; }

    /**
     * @brief Equality comparison between two shapes.
     *
     * @param other shape to compare against
     * @return true if both shapes have the same dimensions and sizes
     */
    bool operator==(Shape const &other) const {
        bool check = true;
        check &= other.numDims() == this->numDims();
        check &= other.isScalar() == this->isScalar();
        if (this->isScalar()) {
            return check;
        }
        for (auto i = 0; i < m_shape.size(); ++i) {
            check &= m_shape[i] == other.m_shape[i];
        }
        return check;
    };

    /** @brief Check if the tensor memory layout is contiguous. Always true for now. */
    bool isContiguous() const { return true; }

    /** @brief Check if this shape represents a scalar (0 dimensions). */
    bool isScalar() const { return m_num_dims == 0; }

 private:
    /**
     * @brief Computes the stride information.
     * Currently only supports contiguous tensors.
     *
     */
    void initialize() {
        /**
         * A 1D tensor of shape 5 has stride = 1
         * 2D tensor (matrix) let say of shape (3,5) will have inner stride 1 and outer stride = 5
         * 3D tensor of shape (2,3,5) will have stride (5*3,5,1)
         *
         * Generalizing shape(C,H,W) will have stride (H*W, W, 1)
         *
         * This assumes tensor is contiguous.
         */

        m_num_dims = m_shape.size();
        if (!isScalar()) {
            m_stride.resize(m_num_dims, 1);
            m_stride[m_num_dims - 1] = 1;
            for (auto i = m_num_dims - 1; i >= 1; --i) {
                m_stride[i - 1] = m_shape[i] * m_stride[i];
            }
        }
    }
    std::vector<size_t> m_shape;
    std::vector<size_t> m_stride;
    size_t m_num_dims;
    std::vector<std::string> m_dim_names;
};

/**
 * @brief Stream insertion operator for printing a Shape.
 *
 * @param os output stream
 * @param shape shape to print
 * @return std::ostream& the output stream
 */
inline std::ostream &operator<<(std::ostream &os, const Shape &shape) {
    os << "[";
    if (shape.isScalar()) {
        os << "scalar";
    } else {
        const auto &vec = shape.shapeVec();
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i)
                os << ", ";
            os << vec[i];
        }
    }
    os << "]";
    return os;
}

/**
 * @brief TensorView is a tensor accessor which does not own the memory but can be used to access the data in the tensor.
 *
 * TODO preprend the operators with a macro which is basically `__host__ __device__` to use with CUDA.
 *
 * @tparam T datatype
 */
template <class T> class TensorView {
 public:
    using value_type = T;                                       ///< datatype
    using reference = value_type &;                             ///< reference type
    using const_reference = const value_type &;                 ///< const reference type
    using pointer = value_type *;                               ///< pointer type
    using const_pointer = const value_type *;                   ///< const pointer type
    using size_type = size_t;                                   ///< size type
    using ptr = typename std::shared_ptr<TensorView<T>>;        ///< shared pointer type
    using unique_ptr = typename std::unique_ptr<TensorView<T>>; ///< unique pointer type

    /** @brief Default construct a null (unbound) view. Must be rebound via operator= before use. */
    TensorView() : m_data(nullptr), m_shape() {}

    /**
     * @brief Construct a TensorView from a raw pointer and shape.
     *
     * @param data pointer to the tensor data (not owned)
     * @param shape shape describing the tensor dimensions
     */
    TensorView(pointer data, const Shape &shape) : m_data(data), m_shape(shape) {}

    /**
     * @brief Copy construct a TensorView (const overload).
     *
     * @param view tensor view to copy from
     */
    TensorView(const TensorView &view) : m_data(view.m_data), m_shape(view.shape()) {}

    /**
     * @brief Copy construct a TensorView (non-const overload).
     *
     * @param view tensor view to copy from
     */
    TensorView(TensorView &view) : m_data(view.m_data), m_shape(view.shape()) {}

    TensorView &operator=(const TensorView &other) {
        m_data = other.m_data;
        m_shape = other.m_shape;
        return *this;
    }

    /**
     * @brief Access an element using multi-dimensional indices.
     *
     * @tparam ARGS index types
     * @param args tensor indices
     * @return reference to the element
     */
    template <typename... ARGS> auto operator()(ARGS... args) -> reference {
        assert(m_data != nullptr);
        return *(m_data + m_shape(args...));
    }

    /**
     * @brief Const access an element using multi-dimensional indices.
     *
     * @tparam ARGS index types
     * @param args tensor indices
     * @return const reference to the element
     */
    template <typename... ARGS> auto operator()(ARGS... args) const -> const_reference {
        assert(m_data != nullptr);
        return *(m_data + m_shape(args...));
    }

    /**
     * @brief Access an element by flat (linear) index.
     *
     * @param index flat index into the underlying data
     * @return reference to the element
     */
    auto operator[](size_t index) -> reference {
        assert(index < m_shape.numElements());
        return *(m_data + index);
    }

    /**
     * @brief Const access an element by flat (linear) index.
     *
     * @param index flat index into the underlying data
     * @return const reference to the element
     */
    auto operator[](size_t index) const -> const_reference {
        assert(index < m_shape.numElements());
        return *(m_data + index);
    }

    /**
     * @brief Get a sub-view (slice) of this tensor along the leading dimensions.
     *
     * @tparam ARGS index types
     * @param idx first dimension index
     * @param args remaining dimension indices
     * @return TensorView<T> a view into the sliced region
     */
    template <typename... ARGS> auto slice(size_t idx, ARGS... args) -> TensorView<T> {
        pointer begin = data();
        size_t offset = m_shape.offset(idx, args...);
        Shape new_shape = m_shape.slice(idx, args...);
        return {begin + offset, new_shape};
    }

    template <typename... ARGS> auto slice(size_t idx, ARGS... args) const -> TensorView<T> {
        size_t offset = m_shape.offset(idx, args...);
        Shape new_shape = m_shape.slice(idx, args...);
        return {const_cast<pointer>(data()) + offset, new_shape};
    }

    /**
     * @brief Return a new view of the same data with a different shape.
     *
     * @param shape new shape (must have the same number of elements)
     * @return TensorView<T> reshaped view
     */
    auto view(const Shape &shape) {
        assert(m_shape.numElements() == shape.numElements());
        return TensorView(this->data(), shape);
    }

    /** @brief Get the shape of this tensor view. */
    auto shape() const -> const Shape & { return m_shape; }

    /** @brief Get the total number of elements. */
    auto size() const -> const size_t { return m_shape.size(); }

    /**
     * @brief Set a new shape for this view.
     *
     * @param shape new shape to assign
     */
    auto setShape(const Shape &shape) { m_shape = shape; }

    /** @brief Get a const pointer to the underlying data. */
    auto data() const -> const_pointer { return m_data; }

    /** @brief Get a mutable pointer to the underlying data. */
    auto data() -> pointer { return m_data; }

    /**
     * @brief Set the underlying data pointer.
     *
     * @param p new data pointer
     */
    auto setData(pointer p) { m_data = p; }

    /** @brief Get the total number of elements. */
    auto numElements() const -> const size_t { return m_shape.numElements(); }

    /** @brief Get the total size in bytes. */
    auto numBytes() const -> const size_t { return numElements() * sizeof(value_type); }

    /** @brief Check if the tensor memory layout is contiguous. */
    auto isContiguous() -> bool { return m_shape.isContiguous(); }

 private:
    pointer m_data;
    Shape m_shape;
};

/**
 * @brief N Dimensional tensor.
 *
 * @tparam COMPUTE device on which the tensor memory is stored.
 * @tparam T datatype.
 */
template <template <class> class COMPUTE, class T> class Tensor : public TensorView<T> {
 public:
    using super = TensorView<T>;
    using value_type = super::value_type;                            ///< datatype
    using reference = value_type &;                                  ///< reference type
    using const_reference = const value_type &;                      ///< const reference type
    using pointer = value_type *;                                    ///< pointer type
    using size_type = size_t;                                        ///< size type
    using ptr = typename std::shared_ptr<Tensor<COMPUTE, T>>;        ///< shared pointer type
    using unique_ptr = typename std::unique_ptr<Tensor<COMPUTE, T>>; ///< unique pointer type

    /** @brief Construct an empty tensor with no shape or data. */
    Tensor() : TensorView<T>(nullptr, Shape()), m_memory({}) {}

    /**
     * @brief Construct a non-owning Tensor that borrows from an existing TensorView.
     *
     * m_memory stays empty; the caller (e.g. TransformerWeights) retains ownership.
     * The source TensorView must outlive this Tensor.
     *
     * @param view the TensorView to borrow from
     */
    Tensor(TensorView<T> view) : TensorView<T>(view), m_memory({}) {}

    /**
     * @brief Borrow from a TensorView without copying data or touching m_memory.
     *
     * Only updates the shape and data pointer; ownership stays with the source.
     * The source TensorView must outlive this Tensor.
     *
     * @param view the TensorView to borrow from
     */
    Tensor &operator=(TensorView<T> view) {
        this->setShape(view.shape());
        this->setData(view.data());
        return *this;
    }
    /**
     * @brief Copy construct a Tensor, deep-copying the data.
     *
     * @param other tensor to copy from
     */
    Tensor(const Tensor &other) : TensorView<T>(nullptr, other.shape()), m_memory(other.m_memory) { this->setData(m_memory.data()); }

    /**
     * @brief Move construct a Tensor, transferring ownership of the buffer.
     *
     * @param other tensor to move from
     */
    Tensor(Tensor &&other) noexcept : TensorView<T>(nullptr, other.shape()), m_memory(std::move(other.m_memory)) {
        this->setData(m_memory.data());
        other.setData(nullptr);
    }

    /**
     * @brief Construct a Tensor by copying data from a raw pointer.
     *
     * @param ptr pointer to source data
     * @param shape shape of the tensor
     */
    Tensor(const pointer ptr, const Shape &shape) : TensorView<T>(nullptr, shape), m_memory(shape.size()) {
        this->setData(m_memory.data());
        copyFrom(ptr, this->numElements());
    }

    /**
     * @brief Construct a Tensor with the given shape (uninitialized data).
     *
     * @param shape shape of the tensor
     */
    Tensor(const Shape &shape) : TensorView<T>(nullptr, shape), m_memory(shape.size()) { this->setData(m_memory.data()); }

    /**
     * @brief Construct a Tensor with the given shape and initial values.
     *
     * @param shape shape of the tensor
     * @param values vector of initial values (size must match shape)
     */
    Tensor(const Shape &shape, std::vector<value_type> &values) : TensorView<T>(nullptr, shape), m_memory(values) { this->setData(m_memory.data()); }

    /**
     * @brief Reshape the tensor, reallocating memory if needed.
     *
     * @param shape new shape
     */
    void reShape(const Shape &shape) {
        this->setShape(shape);
        m_memory.resize(shape.size());
        this->setData(m_memory.data());
    }

    /**
     * @brief Copy data from a raw pointer into this tensor.
     *
     * @param p source pointer
     * @param num_elements number of elements to copy
     */
    void copyFrom(const pointer p, size_t num_elements) { m_memory.copyFrom(p, num_elements); }

    /**
     * @brief Copy data and shape from a TensorView into this tensor.
     *
     * @param tensor source tensor view
     */
    void copyFrom(TensorView<value_type> &tensor) {
        m_memory.copyFrom(tensor.data(), tensor.size());
        this->setShape(tensor.shape());
    }

    /**
     * @brief Copy data and shape from another Tensor (possibly on a different backend).
     *
     * @tparam COMPUTE_OTHER compute backend of the source tensor
     * @tparam T_OTHER data type of the source tensor
     * @param other source tensor to copy from
     */
    template <template <class> class COMPUTE_OTHER, class T_OTHER> void copyFrom(Tensor<COMPUTE_OTHER, T_OTHER> &other) {
        m_memory.copyFrom(other.m_memory);
        this->setShape(other.shape());
    }

 private:
    Memory<COMPUTE, value_type> m_memory; // a memory buffer owned by the tensor.
};

} // namespace transformers_lite