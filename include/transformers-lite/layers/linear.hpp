#pragma once
#include <map>
#include <string>

#include "../core/ops.hpp"
#include "../core/tensor.hpp"
#include "layer.hpp"

namespace transformers_lite {

/**
 * @brief Linear (fully-connected) layer: out = W * x.
 *
 * @tparam COMPUTE compute backend
 * @tparam T datatype
 */
template <template <class> class COMPUTE, class T> class Linear : public LayerBase<COMPUTE, T, Linear<COMPUTE, T>> {
    using Base = LayerBase<COMPUTE, T, Linear<COMPUTE, T>>;

 public:
    using typename Base::compute;
    using typename Base::ptr;
    using typename Base::value_type;

    /** @brief Construct an unbound Linear layer; call initializeLayer before forward. */
    Linear() = default;

    /**
     * @brief Construct a Linear layer with a pre-bound weight view.
     *
     * @param wcls weight matrix view (out_dim, in_dim)
     */
    explicit Linear(TensorView<value_type> &wcls) : m_wcls(wcls) {}

    /**
     * @brief Bind weight views from a state dict.
     *
     * Expected keys: "wcls".
     *
     * @param state_dict map of weight name to tensor view
     */
    void initializeLayer(const std::map<std::string, TensorView<value_type>> &state_dict) { m_wcls = state_dict.at("wcls"); }

    /**
     * @brief Forward pass: out = wcls * x.
     *
     * @param x   input tensor (in_dim)
     * @param out output tensor (out_dim)
     */
    void forward(const Tensor<COMPUTE, value_type> &x, Tensor<COMPUTE, value_type> &out) { matmul(out, x, m_wcls); }

    /** @brief Output dimension (number of rows in the weight matrix). */
    auto outDim() const -> size_t { return m_wcls.shape().shapeVec()[0]; }

 private:
    TensorView<value_type> m_wcls; // (out_dim, in_dim)
};

} // namespace transformers_lite
