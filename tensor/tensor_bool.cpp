#include "tensor.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <string>

Tensor<bool>::Tensor(const std::vector<size_t>& dimensions)
{
    m_shape = dimensions;
    m_size = 1;
    for (auto& v : dimensions)
        m_size *= v;

    if (m_size == 0)
        throw std::runtime_error("Create tensor error: dimensions incorrect");

    m_data.resize(m_size);
    CalculateStrides();
}
Tensor<bool>::Tensor(const std::vector<size_t>& dimensions, bool value) : Tensor(dimensions)
{
    std::fill(m_data.begin(), m_data.end(), value);
}
Tensor<bool>::Tensor(const std::vector<size_t>& dimensions, const std::vector<bool>& values) : Tensor(dimensions)
{
    if (values.size() != m_data.size())
        throw std::runtime_error("Create tensor error: given values count not equal tensor data count");

    m_data = values;
}
Tensor<bool>::Tensor(const std::vector<size_t>& dimensions, const std::vector<size_t>& strides, const std::vector<bool>& values) : Tensor(dimensions)
{
    if (values.size() != m_size)
        throw std::runtime_error("Create tensor error: given values count not equal tensor data count");

    m_data = values;
    m_strides = strides;
}

void Tensor<bool>::CalculateStrides()
{
    m_strides.resize(m_shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

bool Tensor<bool>::at(size_t id)
{
    if (m_shape.size() != 1)
        throw std::runtime_error("Function 'at(size_t)' only for 1-Dimension tensor");

    if (id > m_data.size())
        throw std::out_of_range("Function 'at(size_t)' out of range");

    return m_data[id];
}
bool Tensor<bool>::at(size_t m, size_t n)
{
    if (m_shape.size() != 2)
        throw std::runtime_error("Function 'at(size_t, size_t)' only for 2-Dimension tensor");

    size_t id = m * m_strides[0] + n;
    if (id > m_data.size())
        throw std::out_of_range("Function 'at(size_t, size_t)' out of range");

    return m_data[id];
}
bool Tensor<bool>::at(const std::vector<size_t>& ids)
{
    if (m_shape.size() != ids.size())
        throw std::runtime_error("Function 'at(const std::vector<size_t>&)' incorrect dimension. Tensor dimension = " + std::to_string(m_shape.size()) + "; IDs dimension = " + std::to_string(ids.size()));

    size_t id = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] > m_shape[i])
            throw std::out_of_range("Function 'at(const std::vector<size_t>&)' out of range");

        id += ids[i] * m_strides[i];
    }

    if (id > m_data.size())
        throw std::out_of_range("Function 'at(size_t, size_t)' out of range");

    return m_data[id];
}

auto Tensor<bool>::Shape()   const -> const std::vector<size_t>& { return m_shape;   }
auto Tensor<bool>::Strides() const -> const std::vector<size_t>& { return m_strides; }
auto Tensor<bool>::Data()    const -> const std::vector<bool>&   { return m_data;    }
auto Tensor<bool>::Data()          -> std::vector<bool>&         { return m_data;    }
auto Tensor<bool>::Size()    const -> size_t                     { return m_size;    }

Tensor<bool> Tensor<bool>::operator==(const Tensor<bool>& other) const 
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_data.size() != other.m_data.size()) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.m_data[i] = (m_data[i] == other.m_data[i]);

    return result;
}
Tensor<bool> Tensor<bool>::operator!=(const Tensor<bool>& other) const 
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_data.size() != other.m_data.size()) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.m_data[i] = (m_data[i] != other.m_data[i]);

    return result;
}