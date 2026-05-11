#pragma once
#include <algorithm>
#include <iostream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace transformers_lite {

/**
 * @brief Flat weight dictionary with PyTorch-style dot-notation keys and prefix-based slicing.
 *
 * @tparam T element type of the stored TensorViews
 */
template <typename T> class StateDict {
 public:
    using map_type = std::unordered_map<std::string, TensorView<T>>;

    StateDict() = default;

    StateDict(std::initializer_list<typename map_type::value_type> initList) : m_map(initList) {}

    explicit StateDict(map_type map) : m_map(std::move(map)) {}

    TensorView<T> &operator[](const std::string &key) { return m_map[key]; }

    const TensorView<T> &at(const std::string &key) const { return m_map.at(key); }
    TensorView<T> &at(const std::string &key) { return m_map.at(key); }

    /**
     * @brief Return a sub-dict of keys matching "prefix.*" with the prefix stripped.
     *
     * E.g. getLayerWeights("layers.0") on {"layers.0.attention.wq.weight": v}
     * returns {"attention.wq.weight": v}.
     *
     * @param prefix dot-separated key prefix to match and strip
     */
    StateDict<T> getLayerWeights(const std::string &prefix) const {
        const std::string prefix_dot = prefix + ".";
        map_type result;
        for (const auto &[key, val] : m_map) {
            if (key.starts_with(prefix_dot))
                result.emplace(key.substr(prefix_dot.size()), val);
        }
        return StateDict<T>(std::move(result));
    }

    /** @brief Print all keys and their tensor shapes to os, sorted alphabetically. */
    void print(std::ostream &out = std::cout) const {
        std::vector<std::string> keys;
        keys.reserve(m_map.size());
        for (const auto &[key, _] : m_map) {
            keys.push_back(key);
        }
        std::ranges::sort(keys);
        for (const auto &key : keys) {
            out << key << ": " << m_map.at(key).shape() << "\n";
        }
    }

 private:
    map_type m_map;
};

} // namespace transformers_lite
