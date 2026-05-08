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
template <template <class> class COMPUTE, class T> class Memory {
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
     * @param num_elements number of elements in the memory
     */
    Memory(const size_t num_elements) : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0) {
        reserve(num_elements);
        // resize(num_elements);
    }

    /**
     * @brief Construct a new Memory object
     *
     * @param scalar initialize memory with scalar value
     * @param num_elements number of elements in the memory
     */
    Memory(const T scalar, const size_t num_elements) : m_alloc() {
        // resize(num_elements);
        reserve(num_elements);
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
    Memory(const Memory &other) : m_alloc() {
        reserve(other.size());
        COMPUTE<T>::copy(other.data(), m_data, other.size());
    }

    /**
     * @brief Copy construct a new Memory object (non-const overload).
     *
     * @param other memory to copy from
     */
    Memory(Memory &other) : m_alloc() {
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
    virtual ~Memory() { m_alloc.deallocate(m_data, m_allocated_size); }

    /**
     * @brief resizes the memory buffer
     *
     * @param num_elements
     */
    void resize(const size_t num_elements, value_type val = value_type()) {
        // TODO need to copy data in resize.
        if (num_elements == m_allocated_size) {
            return;
        }
        if (num_elements > m_allocated_size) {
            auto temp = m_alloc.allocate(num_elements);
            if (m_data != nullptr) {
                COMPUTE<T>::copy(m_data, temp, m_size);
            }
            COMPUTE<T>::fill(temp + m_size, num_elements - m_size, val);
            if (m_data != nullptr) {
                m_alloc.deallocate(m_data, m_allocated_size);
            }
            m_data = temp;
            m_allocated_size = num_elements;
            m_size = num_elements;
        } else // num_elements less than m_size
        {
            m_size = num_elements;
        }
    }

    /**
     * @brief allocates memory for given number of elements
     *
     * @param num_elements
     */
    void reserve(const size_t num_elements) {
        if (num_elements > m_allocated_size) {
            m_data = m_alloc.allocate(num_elements);
            m_allocated_size = num_elements;
        }
        m_size = num_elements;
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

    /** @brief Get a const pointer to the underlying data. */
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
     * @param num_elements number of elements to copy
     */
    auto copyFrom(pointer data, size_t num_elements) { COMPUTE<T>::copy(data, m_data, num_elements); }

    /**
     * @brief Copy data from another Memory buffer into this buffer.
     *
     * @tparam COMPUTE_OTHER compute backend of the source memory
     * @tparam T_OTHER data type of the source memory
     * @param other_mem source memory to copy from
     */
    template <template <class> class COMPUTE_OTHER, class T_OTHER> auto copyFrom(Memory<COMPUTE_OTHER, T_OTHER> &other_mem) {
        COMPUTE<T>::copy(other_mem.data(), m_data, other_mem.size());
    }

    // TODO: implement iterators.

 private:
    allocator_type m_alloc;
    pointer m_data;
    size_type m_size;
    size_type m_allocated_size;
};

} // namespace transformers_lite