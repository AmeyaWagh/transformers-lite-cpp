#pragma once
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "compute.hpp"

namespace transformers_lite {

/**
 * @brief A generic memory buffer
 * @TODO make this cuda compatable.
 *
 * @tparam T datatype
 * @tparam Alloc memory allocator for the buffer
 */
template <template <class> class COMPUTE, class T>
requires ComputeBackend<COMPUTE<T>, T>
class Memory {
 public:
    using allocator_type = COMPUTE<T>::allocator_type;               // allocator type
    using value_type = T;                                            // datatype
    using reference = value_type &;                                  // reference type
    using const_reference = const value_type &;                      // const reference type
    using pointer = value_type *;                                    // pointer type
    using const_pointer = const value_type *;                        // const pointer type
    using size_type = size_t;                                        // size type
    using ptr = typename std::shared_ptr<Memory<COMPUTE, T>>;        // shared pointer type
    using unique_ptr = typename std::unique_ptr<Memory<COMPUTE, T>>; // unique pointer type

    /**
     * @brief Construct a new Memory object
     *
     */
    Memory() : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0) {}

    /**
     * @brief Construct a new Memory object
     *
     * @param numElements number of elements in the memory
     */
    Memory(const size_t numElements) : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0) {
        reserve(numElements);
        // resize(numElements);
    }

    /**
     * @brief Construct a new Memory object
     *
     * @param scalar initialize memory with scalar value
     * @param numElements number of elements in the memory
     */
    Memory(const T scalar, const size_t numElements) : m_alloc() {
        // resize(numElements);
        reserve(numElements);
        COMPUTE<T>::fill(m_data, m_size, scalar);
    }

    /**
     * @brief Construct a new Memory object from a vector of values.
     *
     * @param values vector of values to copy into the memory
     */
    Memory(const std::vector<value_type> &values) : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0) {
        reserve(values.size());
        COMPUTE<T>::copy(values.data(), m_data, values.size());
        m_size = values.size();
        m_allocated_size = values.size();
    }

    /**
     * @brief Copy Construct a new Memory object
     *
     * @param other
     */
    Memory(const Memory &other) : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0) {
        reserve(other.size());
        COMPUTE<T>::copy(other.data(), m_data, other.size());
    }

    /**
     * @brief Copy construct a new Memory object (non-const overload).
     *
     * @param other memory to copy from
     */
    Memory(Memory &other) : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0) {
        reserve(other.size());
        COMPUTE<T>::copy(other.data(), m_data, other.size());
    }

    /**
     * @brief Move construct a Memory object, transferring ownership.
     *
     * @param other memory to move from
     */
    Memory(Memory &&other) noexcept : m_alloc(), m_data(other.m_data), m_size(other.m_size), m_allocated_size(other.m_allocated_size) {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_allocated_size = 0;
    }

    /** @brief Destroy the Memory object and deallocate the buffer. */
    virtual ~Memory() {
        if (m_data != nullptr)
            m_alloc.deallocate(m_data, m_allocated_size);
    }

    /**
     * @brief resizes the memory buffer
     *
     * @param numElements
     */
    void resize(const size_t numElements, value_type val = value_type()) {
        // TODO need to copy data in resize.
        if (numElements == m_allocated_size) {
            return;
        }
        if (numElements > m_allocated_size) {
            auto temp = m_alloc.allocate(numElements);
            if (m_data != nullptr) {
                COMPUTE<T>::copy(m_data, temp, m_size);
            }
            COMPUTE<T>::fill(temp + m_size, numElements - m_size, val);
            if (m_data != nullptr) {
                m_alloc.deallocate(m_data, m_allocated_size);
            }
            m_data = temp;
            m_allocated_size = numElements;
            m_size = numElements;
        } else // numElements less than m_size
        {
            m_size = numElements;
        }
    }

    /**
     * @brief allocates memory for given number of elements
     *
     * @param numElements
     */
    void reserve(const size_t numElements) {
        if (numElements > m_allocated_size) {
            m_data = m_alloc.allocate(numElements);
            m_allocated_size = numElements;
        }
        m_size = numElements;
    }

    /**
     * @brief access memory at the given index
     *
     * @param index index to the element in the memory
     * @return T& reference to the element pointed by the index in the memory
     */
    auto operator[](size_t index) -> reference {
        if (index >= m_size) {
            throw std::runtime_error("Array index out of bound");
        }
        return COMPUTE<T>::get(m_data, index);
    }

    /**
     * @brief Const access memory at the given index.
     *
     * @param index index to the element in the memory
     * @return const T& const reference to the element
     */
    auto operator[](size_t index) const -> const_reference {
        if (index >= m_size) {
            throw std::runtime_error("Array index out of bound");
        }
        return COMPUTE<T>::get(m_data, index);
    }

    /**
     * @brief Current size of the memory
     *
     * @return size_t
     */
    auto size() const -> size_type { return m_size; }

    /**
     * @brief raw pointer to the memory data
     *
     * @return T* pointer
     */
    auto data() -> pointer { return m_data; }

    /** @brief Get a const pointer to the underlying data.
     *
     * @return const T* const pointer to the data
     */
    auto data() const -> const_pointer { return m_data; }

    /**
     * @brief Check if the memory buffer is empty.
     *
     * @return true if size is 0
     */
    auto empty() -> bool { return size() == 0; }

    /**
     * @brief Copy data from a raw pointer into this buffer.
     *
     * @param data source pointer
     * @param numElements number of elements to copy
     */
    auto copyFrom(const_pointer data, size_t numElements) { COMPUTE<T>::copy(data, m_data, numElements); }

    /**
     * @brief Copy data from another Memory buffer into this buffer.
     *
     * @tparam COMPUTE_OTHER compute backend of the source memory
     * @tparam T_OTHER data type of the source memory
     * @param otherMem source memory to copy from
     */
    template <template <class> class COMPUTE_OTHER, class T_OTHER> auto copyFrom(Memory<COMPUTE_OTHER, T_OTHER> &otherMem) {
        COMPUTE<T>::copy(otherMem.data(), m_data, otherMem.size());
    }

    // TODO: implement iterators.

 private:
    allocator_type m_alloc;
    pointer m_data;
    size_type m_size;
    size_type m_allocated_size;
};

} // namespace transformers_lite