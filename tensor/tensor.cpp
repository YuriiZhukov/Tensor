#include "tensor.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <iostream>
#include <math.h>
#include <stdexcept>

#include "random_generator.h"

#define TMP template<typename T>

TMP Tensor<T>::Tensor(const std::vector<size_t>& dimensions)
{
    m_shape = dimensions;
    m_size = 1;
    for (auto& v : dimensions) 
        m_size *= v;

    if (m_size == 0) 
        throw std::runtime_error("Error: incorrect dimensions");
    
    m_data.resize(m_size);
    CalculateStrides();
}
TMP Tensor<T>::Tensor(const std::vector<size_t>& dimensions, const T& value) : Tensor(dimensions)
{
    std::fill(m_data.begin(), m_data.end(), value);
}
TMP Tensor<T>::Tensor(const std::vector<size_t>& dimensions, const std::vector<T>& values) : Tensor(dimensions)
{
    if (values.size() != m_size)
        throw std::runtime_error("Error: given values count not equal tensor data count");

    m_data = values;
}
TMP Tensor<T>::Tensor(const std::vector<size_t>& dimensions, const std::vector<size_t>& strides, const std::vector<T>& values) : Tensor(dimensions)
{
    if (values.size() != m_size)
        throw std::runtime_error("Error: given values count not equal tensor data count");

    m_data = values;
    m_strides = strides;
}
TMP Tensor<T>::Tensor(const T& value) : Tensor(std::vector<size_t>{})
{
    m_data[0] = value;
}

TMP void Tensor<T>::CalculateStrides()
{
    if (m_shape.size() == 0)
        return;

    m_strides.resize(m_shape.size());
    m_strides.back() = 1;
    for (int i = m_shape.size() - 2; i >= 0; --i) {
        m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
    }
}

TMP T& Tensor<T>::At(size_t id)
{
    if (m_shape.size() != 1)
        throw std::runtime_error("Function 'At(size_t)' only for 1-Dimension tensor");

    if (id > m_size)
        throw std::out_of_range("Function 'At(size_t)' out of range");

    return m_data[id];
}
TMP T& Tensor<T>::At(size_t m, size_t n)
{
    if (m_shape.size() != 2)
        throw std::runtime_error("Function 'At(size_t, size_t)' only for 2-Dimension tensor");

    size_t id = m * m_strides[0] + n;
    if (id > m_size)
        throw std::out_of_range("Function 'At(size_t, size_t)' out of range");

    return m_data[id];
}
TMP T& Tensor<T>::At(const std::vector<size_t>& ids)
{
    if (m_shape.size() != ids.size())
        throw std::runtime_error("Function 'At(const std::vector<size_t>&)' incorrect dimension. Tensor dimension = " + std::to_string(m_shape.size()) + "; IDs dimension = " + std::to_string(ids.size()));

    size_t id = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] > m_shape[i])
            throw std::out_of_range("Function 'At(const std::vector<size_t>&)' out of range");

        id += ids[i] * m_strides[i];
    }

    if (id > m_size)
        throw std::out_of_range("Function 'At(size_t, size_t)' out of range");

    return m_data[id];
}

TMP const T& Tensor<T>::At(size_t id) const
{
    if (m_shape.size() != 1)
        throw std::runtime_error("Function 'At(size_t)' only for 1-Dimension tensor");

    if (id > m_size)
        throw std::out_of_range("Function 'At(size_t)' out of range");

    return m_data[id];
}
TMP const T& Tensor<T>::At(size_t m, size_t n) const
{
    if (m_shape.size() != 2)
        throw std::runtime_error("Function 'At(size_t, size_t)' only for 2-Dimension tensor");

    size_t id = m * m_strides[0] + n;
    if (id > m_size)
        throw std::out_of_range("Function 'At(size_t, size_t)' out of range");

    return m_data[id];
}
TMP const T& Tensor<T>::At(const std::vector<size_t>& ids) const
{
    if (m_shape.size() != ids.size())
        throw std::runtime_error("Function 'At(const std::vector<size_t>&)' incorrect dimension. Tensor dimension = " + std::to_string(m_shape.size()) + "; IDs dimension = " + std::to_string(ids.size()));

    size_t id = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] > m_shape[i])
            throw std::out_of_range("Function 'At(const std::vector<size_t>&)' out of range");

        id += ids[i] * m_strides[i];
    }

    if (id > m_size)
        throw std::out_of_range("Function 'At(size_t, size_t)' out of range");

    return m_data[id];
}

TMP auto Tensor<T>::Shape() const -> const std::vector<size_t>&
{
    return m_shape;   
}
TMP auto Tensor<T>::Strides() const -> const std::vector<size_t>&
{
    return m_strides; 
}
TMP auto Tensor<T>::Data() const -> const std::vector<T>&
{
    return m_data;    
}
TMP auto Tensor<T>::Size() const -> size_t
{
    return m_size;    
}

TMP auto Tensor<T>::Shape() -> std::vector<size_t>&
{
    return m_shape;
}
TMP auto Tensor<T>::Strides() -> std::vector<size_t>&
{
    return m_strides;
}
TMP auto Tensor<T>::Data() -> std::vector<T>&
{
    return m_data;
}

TMP Tensor<T> Tensor<T>::Zeros(const std::vector<size_t>& dimensions) 
{
    return Tensor<T>(dimensions, 0);
}
TMP Tensor<T> Tensor<T>::Ones(const std::vector<size_t>& dimensions)
{
    return Tensor<T>(dimensions, 1);
}
TMP Tensor<T> Tensor<T>::Full(const std::vector<size_t>& dimensions, const T& value) 
{
    return Tensor<T>(dimensions, value);
}
TMP Tensor<T> Tensor<T>::Eye(size_t size) 
{
    return Tensor<T>({size, size}, 1);
}
TMP Tensor<T> Tensor<T>::Random(const std::vector<size_t>& dimensions, const T& min, const T& max) 
{
    RandomGenerator<T> rnd;
    size_t size = 1;
    for (const auto& val : dimensions)
        size *= val;
    Tensor<T> tensor(dimensions, rnd.Get(min, max, size));
    return tensor;
}
TMP Tensor<T> Tensor<T>::Arange(const T& start, const T& stop, const T& step) 
{
    size_t num = (stop - start) / step;
    std::vector<T> values;
    values.resize(num);
    for (size_t i = 0; i < num; ++i)
        values[i] = start + i * step;

    return Tensor<T>({ num }, values);
}
TMP Tensor<T> Tensor<T>::Linspace(const T& start, const T& stop, size_t num)
{
    T step = (stop - start) / static_cast<T>(num);
    std::vector<T> values;
    values.resize(num);
    for (size_t i = 0; i < num; ++i)
        values[i] = start + i * step;

    return Tensor<T>({ num }, values);
}

TMP Tensor<T>& Tensor<T>::Reshape_Inplace(const std::vector<size_t>& dimensions)
{
    size_t new_size = 1;
    for (auto d : dimensions)
        new_size *= d;

    if (new_size != m_size)
        throw std::runtime_error("Cannot reshape tenzor. Result elements count not equal current elements count");

    m_shape = dimensions;
    CalculateStrides();
    return *this;
}
TMP Tensor<T>& Tensor<T>::Flatten_Inplace()
{
    m_shape = { m_size };
    CalculateStrides();
    return *this;
}
TMP Tensor<T>& Tensor<T>::Transpose_Inplace()
{
    if (m_shape.size() != 2)
        throw std::runtime_error("Cannot transpose tenzor. Tensor dimensions not equal 2. Overload 'Transpose(size_t, size_t)' need to be used");

    std::swap(m_shape[1], m_shape[0]);
    std::swap(m_strides[1], m_strides[0]);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Transpose_Inplace(size_t dim1, size_t dim2)
{
    if (dim1 >= m_shape.size() || dim2 >= m_shape.size())
        throw std::runtime_error("Cannot transpose tenzor. Dim1 or dim2 greater than shape size");

    std::swap(m_shape[dim2], m_shape[dim1]);
    std::swap(m_strides[dim2], m_strides[dim1]);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Permute_Inplace(const std::vector<size_t>& axis)
{
    if (axis.size() != m_shape.size())
        throw std::runtime_error("Cannot permute tenzor. Axis count not equal current shape size");

    size_t size = axis.size();

    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    new_shape.resize(size);
    new_strides.resize(size);
    for (size_t i = 0; i < size; ++i) {
        if (axis[i] < m_shape.size()) {
            new_shape[i] = m_shape[axis[i]];
            new_strides[i] = m_strides[axis[i]];
        }
        else throw std::runtime_error("Cannot permute tenzor. Axis out of range");
    }

    m_shape = new_shape;
    m_strides = new_strides;
    return *this;
}


TMP Tensor<T> Tensor<T>::Squeeze(std::vector<size_t> dimensions)
{
    for (auto& d : dimensions) {
        if (d >= m_shape.size()) {
            throw std::runtime_error(std::string(__func__) + " error: Input dimension out of range");
        }
    }

    std::vector<size_t> new_shape = m_shape;
    std::vector<size_t> sort_dims = dimensions;
    std::sort(sort_dims.begin(), sort_dims.end(), std::greater<size_t>());

    for (auto& d : dimensions) {
        if (new_shape[d] == 1) {
            new_shape.erase(new_shape.begin() + d);
        }
    }
    
    return this->Reshape(new_shape);

}
TMP Tensor<T> Tensor<T>::Squeeze(size_t dimension)
{
    return this->Squeeze(std::vector<size_t>{dimension});
}
TMP Tensor<T> Tensor<T>::Squeeze()
{
    std::vector<size_t> new_shape{};

    for (auto& s : m_shape) {
        if (s != 1) {
            new_shape.push_back(s);
        }
    }

    return this->Reshape(new_shape);
}
TMP Tensor<T> Tensor<T>::Unsqueeze(std::vector<size_t> dimensions)
{
    for (auto& d : dimensions) {
        if (d > m_shape.size()) {
            throw std::runtime_error(std::string(__func__) + " error: Input dimension out of range");
        }
    }

    std::vector<size_t> new_shape = m_shape;
    std::vector<size_t> sort_dims = dimensions;
    std::sort(sort_dims.begin(), sort_dims.end(), std::greater<size_t>());

    for (auto& d : sort_dims)
        new_shape.insert(new_shape.begin() + d, 1);

    return this->Reshape(new_shape);
}

TMP Tensor<T> Tensor<T>::Unsqueeze(size_t dimension)
{
    return this->Unsqueeze(std::vector<size_t>{dimension});
}

TMP Tensor<T> Tensor<T>::Reshape(const std::vector<size_t>& dimensions) const
{
    size_t new_size = 1;
    for (auto d : dimensions)
        new_size *= d;

    if (new_size != m_size) 
        throw std::runtime_error("Cannot reshape tenzor. Result elements count not equal current elements count");
    
    return Tensor<T>(dimensions, m_data);
}
TMP Tensor<T> Tensor<T>::Flatten() const
{
    return Tensor<T>({ m_size }, m_data);
}
TMP Tensor<T> Tensor<T>::Transpose() const
{
    if (m_shape.size() != 2) 
        throw std::runtime_error("Cannot transpose tenzor. Tensor dimensions not equal 2. Overload 'Transpose(size_t, size_t)' need to be used");
    
    return Tensor<T>({ m_shape[1], m_shape[0] }, { m_strides[1], m_strides[0] }, m_data);
}
TMP Tensor<T> Tensor<T>::Transpose(size_t dim1, size_t dim2) const
{
    if (dim1 >= m_shape.size() || dim2 >= m_shape.size())
        throw std::runtime_error("Cannot transpose tenzor. Dim1 or dim2 greater than shape size");

    std::vector<size_t> new_shape = m_shape;
    std::vector<size_t> new_strides = m_strides;
    std::swap(new_shape[dim2], new_shape[dim1]);
    std::swap(new_strides[dim2], new_strides[dim1]);

    return Tensor<T>(new_shape, new_strides, m_data);
}
TMP Tensor<T> Tensor<T>::Permute(const std::vector<size_t>& axis) const
{
    if (axis.size() != m_shape.size())
        throw std::runtime_error("Cannot permute tenzor. Axis count not equal current shape size");

    size_t size = axis.size();

    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    new_shape.resize(size);
    new_strides.resize(size);
    for (size_t i = 0; i < size; ++i) {
        if (axis[i] < m_shape.size() || axis[i] < m_strides.size()) {
            new_shape[i] = m_shape[axis[i]];
            new_strides[i] = m_strides[axis[i]];
        }
        else throw std::runtime_error("Cannot permute tenzor. Axis out of range");
    }

    return Tensor<T>(new_shape, new_strides, m_data);
}

template <typename T> 
T abs_impl(const T& val)
{
    return (val < 0 ? -val : val);
}

TMP Tensor<T>& Tensor<T>::Abs_Inplace()
{
    for (auto& val : m_data)
        val = abs_impl(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Neg_Inplace()
{
    for (auto& val : m_data)
        val = -val;
    return *this;
}
TMP Tensor<T>& Tensor<T>::Exp_Inplace()
{
    for (auto& val : m_data)
        val = std::exp(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Log_Inplace()
{
    for (auto& val : m_data)
        val = std::log(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Log10_Inplace()
{
    for (auto& val : m_data)
        val = std::log10(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Sqrt_Inplace()
{
    for (auto& val : m_data)
        val = std::sqrt(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Rsqrt_Inplace()
{
    for (auto& val : m_data)
        val = T(1) / sqrt(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Sin_Inplace()
{
    for (auto& val : m_data)
        val = std::sin(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Cos_Inplace()
{
    for (auto& val : m_data)
        val = std::cos(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Tan_Inplace()
{
    for (auto& val : m_data)
        val = std::tan(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Floor_Inplace()
{
    for (auto& val : m_data)
        val = std::floor(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Ceil_Inplace()
{
    for (auto& val : m_data)
        val = std::ceil(val);
    return *this;
}
TMP Tensor<T>& Tensor<T>::Round_Inplace()
{
    for (auto& val : m_data)
        val = std::round(val);
    return *this;
}

TMP Tensor<T> Tensor<T>::Abs() const
{
    Tensor<T> result = *this;
    return result.Abs_Inplace();
}
TMP Tensor<T> Tensor<T>::Neg() const
{
    Tensor<T> result = *this;
    return result.Neg_Inplace();
}
TMP Tensor<T> Tensor<T>::Exp() const
{
    Tensor<T> result = *this;
    return result.Exp_Inplace();
}
TMP Tensor<T> Tensor<T>::Log() const
{
    Tensor<T> result = *this;
    return result.Log_Inplace();
}
TMP Tensor<T> Tensor<T>::Log10() const
{
    Tensor<T> result = *this;
    return result.Log10_Inplace();
}
TMP Tensor<T> Tensor<T>::Sqrt() const
{
    Tensor<T> result = *this;
    return result.Sqrt_Inplace();
}
TMP Tensor<T> Tensor<T>::Rsqrt() const
{
    Tensor<T> result = *this;
    return result.Rsqrt_Inplace();
}
TMP Tensor<T> Tensor<T>::Sin() const
{
    Tensor<T> result = *this;
    return result.Sin_Inplace();
}
TMP Tensor<T> Tensor<T>::Cos() const
{
    Tensor<T> result = *this;
    return result.Cos_Inplace();
}
TMP Tensor<T> Tensor<T>::Tan() const
{
    Tensor<T> result = *this;
    return result.Tan_Inplace();
}
TMP Tensor<T> Tensor<T>::Floor() const
{
    Tensor<T> result = *this;
    return result.Floor_Inplace();
}
TMP Tensor<T> Tensor<T>::Ceil() const
{
    Tensor<T> result = *this;
    return result.Ceil_Inplace();
}
TMP Tensor<T> Tensor<T>::Round() const
{
    Tensor<T> result = *this;
    return result.Round_Inplace();
}

TMP bool Tensor<T>::All() const
{
    for (auto& val : m_data) {
        if (!val)
            return false;
    }
    return true;
}
TMP bool Tensor<T>::Any() const
{
    for (auto& val : m_data) {
        if (val)
            return true;
    }
    return false;
}

TMP Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other)
{
    if (other.m_shape != m_shape) {
        std::cout << "Sub tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }
    for (size_t i = 0; i < m_size; ++i)
        m_data[i] += other.m_data[i];

    return *this;
}
TMP Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other)
{
    if (other.m_shape != m_shape) {
        std::cout << "Sub tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }
    for (size_t i = 0; i < m_size; ++i)
        m_data[i] -= other.m_data[i];

    return *this;
}
TMP Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other)
{
    if (other.m_shape != m_shape) {
        std::cout << "Sub tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }
    for (size_t i = 0; i < m_size; ++i)
        m_data[i] *= other.m_data[i];

    return *this;
}
TMP Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other)
{
    if (m_shape != m_shape) {
        std::cout << "Sub tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }
    for (size_t i = 0; i < m_size; ++i) {
        if (other.m_data[i] == T(0)) {
            throw std::runtime_error(std::string("Division by zero in ") + __func__);
            return *this;
        }
        m_data[i] /= other.m_data[i];
    }

    return *this;
}
TMP Tensor<T>& Tensor<T>::operator+=(const T& val)
{
    for (size_t i = 0; i < m_size; ++i)
        m_data[i] += val;

    return *this;
}
TMP Tensor<T>& Tensor<T>::operator-=(const T& val)
{
    for (size_t i = 0; i < m_size; ++i)
        m_data[i] -= val;

    return *this;
}
TMP Tensor<T>& Tensor<T>::operator*=(const T& val)
{
    for (size_t i = 0; i < m_size; ++i)
        m_data[i] *= val;

    return *this;
}
TMP Tensor<T>& Tensor<T>::operator/=(const T& val)
{
    if (val == 0) {
        throw std::runtime_error(std::string("Division by zero in ") + __func__);
        return *this;
    }

    for (size_t i = 0; i < m_size; ++i)
        m_data[i] /= val;

    return *this;
}

TMP Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const
{
    if (other.m_shape != m_shape) {
        std::cout << "Operator '+' tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }

    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i) 
        result.m_data[i] = m_data[i] + other.m_data[i];

    return result;
}
TMP Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const
{
    if (other.m_shape != m_shape) {
        std::cout << "Operator '-' tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }

    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i)
        result.m_data[i] = m_data[i] - other.m_data[i];

    return result;
}
TMP Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const
{
    if (other.m_shape != m_shape) {
        std::cout << "Operator '*' tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }

    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i)
        result.m_data[i] = m_data[i] * other.m_data[i];

    return result;
}
TMP Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const
{
    if (other.m_shape != m_shape) {
        std::cout << "Operator '/' tensor error. Tensor shapes not equal" << std::endl;
        return *this;
    }

    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i) {
        if (other.m_data[i] == T(0))
            throw std::runtime_error(std::string("Division by zero in ") + __func__);
        result.m_data[i] = m_data[i] / other.m_data[i];
    }

    return result;
}
TMP Tensor<T> Tensor<T>::operator+(const T& val) const
{
    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i)
        result.m_data[i] = m_data[i] + val;

    return result;
}
TMP Tensor<T> Tensor<T>::operator-(const T& val) const
{
    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i)
        result.m_data[i] = m_data[i] - val;

    return result;
}
TMP Tensor<T> Tensor<T>::operator*(const T& val) const
{
    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i)
        result.m_data[i] = m_data[i] * val;

    return result;
}
TMP Tensor<T> Tensor<T>::operator/(const T& val) const
{
    if (val == T(0))
        throw std::runtime_error(std::string("Division by zero in ") + __func__);

    Tensor<T> result(m_shape);
    for (size_t i = 0; i < m_size; ++i)
        result.m_data[i] = m_data[i] / val;

    return result;
}

TMP Tensor<T>& Tensor<T>::Add_Inplace(const Tensor<T>& other)
{
    *this += other;
    return *this;
}
TMP Tensor<T>& Tensor<T>::Sub_Inplace(const Tensor<T>& other)
{
    *this -= other;
    return *this;
}
TMP Tensor<T>& Tensor<T>::Mul_Inplace(const Tensor<T>& other)
{
    *this *= other;
    return *this;
}
TMP Tensor<T>& Tensor<T>::Div_Inplace(const Tensor<T>& other)
{
    *this /= other;
    return *this;
}

TMP Tensor<T> Tensor<T>::Add(const Tensor<T>& other) const
{
    return *this + other;
}
TMP Tensor<T> Tensor<T>::Sub(const Tensor<T>& other) const
{
    return *this - other;
}
TMP Tensor<T> Tensor<T>::Mul(const Tensor<T>& other) const
{
    return *this * other;
}
TMP Tensor<T> Tensor<T>::Div(const Tensor<T>& other) const
{
    return *this / other;
}

TMP Tensor<bool> Tensor<T>::operator==(const Tensor<T>& other) const
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_size != other.m_size) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] = (m_data[i] == other.m_data[i]);

    return result;
}
TMP Tensor<bool> Tensor<T>::operator!=(const Tensor<T>& other) const
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_size != other.m_size) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] = (m_data[i] != other.m_data[i]);

    return result;
}
TMP Tensor<bool> Tensor<T>::operator>(const Tensor<T>& other) const
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_size != other.m_size) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] = (m_data[i] > other.m_data[i]);

    return result;
}
TMP Tensor<bool> Tensor<T>::operator<(const Tensor<T>& other) const
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_size != other.m_size) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] = (m_data[i] < other.m_data[i]);

    return result;
}
TMP Tensor<bool> Tensor<T>::operator>=(const Tensor<T>& other) const
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_size != other.m_size) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] = (m_data[i] >= other.m_data[i]);

    return result;
}
TMP Tensor<bool> Tensor<T>::operator<=(const Tensor<T>& other) const
{
    if (m_shape != other.m_shape) {
        std::cout << "Tensor compare error: shapes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    if (m_size != other.m_size) {
        std::cout << "Tensor compare error: sizes not equal" << std::endl;
        return Tensor<bool>({0});
    }
    Tensor<bool> result(m_shape);
    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] = (m_data[i] <= other.m_data[i]);

    return result;
}

TMP T Tensor<T>::Sum() const
{
    if (m_data.empty()) {
        std::cout << "Sum error: Tensor empty" << std::endl;
        return T(0);
    }
    T result = 0;
    for (size_t i = 0; i < m_size; ++i)
        result += m_data[i];

    return result;
}
TMP T Tensor<T>::Prod() const
{
    if (m_data.empty()) {
        std::cout << "Prod error: Tensor empty" << std::endl;
        return T(0);
    }
    T result = 1;
    for (size_t i = 0; i < m_size; ++i)
        result *= m_data[i];

    return result;
}
TMP T Tensor<T>::Mean() const
{
    if (m_data.empty()) {
        std::cout << "Mean error: Tensor empty" << std::endl;
        return T(0);
    }
    T sum = 0;
    for (size_t i = 0; i < m_size; ++i)
        sum += m_data[i];

    return sum / m_size;
}
TMP T Tensor<T>::Var() const
{
    if (m_data.empty()) {
        std::cout << "Variance error: Tensor empty" << std::endl;
        return T(0);
    }

    T mean = Mean();
    T sq_sum = T(0);
    for (auto& val : m_data) {
        T diff = val - mean;
        sq_sum += (diff * diff);
    }

    return sq_sum / m_size;
}
TMP T Tensor<T>::Std() const
{
    if (m_data.empty()) {
        std::cout << "Standard deviation error: Tensor empty" << std::endl;
        return T(0);
    }

    T variance = Var();
    return std::sqrt(variance);
}
TMP T Tensor<T>::Min() const
{
    if (m_data.empty()) {
        std::cout << "Min error: Tensor empty" << std::endl;
        return T(0);
    }
    T min = m_data[0];
    for (size_t i = 0; i < m_size; ++i) {
        if (m_data[i] < min) {
            min = m_data[i];
        }
    }
    return min;
}
TMP T Tensor<T>::Max() const
{
    if (m_data.empty()) {
        std::cout << "Max error: Tensor empty" << std::endl;
        return T(0);
    }
    T max = m_data[0];
    for (size_t i = 0; i < m_size; ++i) {
        if (m_data[i] > max) {
            max = m_data[i];
        }
    }
    return max;
}

TMP Tensor<T> Tensor<T>::Pow_Inplace(const Tensor<T>& exp)
{
    if (m_shape != exp.m_shape)
        throw std::runtime_error(std::string(__func__) + " error: Tensor shapes not equal");

    for (size_t i = 0; i < m_size; ++i)
        m_data[i] = std::pow(m_data[i], exp.m_data[i]);

    return *this;
}
TMP Tensor<T> Tensor<T>::Pow_Inplace(const T& exp)
{
    for (size_t i = 0; i < m_size; ++i)
        m_data[i] = std::pow(m_data[i], exp);

    return *this;
}

TMP Tensor<T> Tensor<T>::Pow(const Tensor<T>& base, const Tensor<T>& exp)
{
    if (base.m_shape != exp.m_shape)
        throw std::runtime_error(std::string(__func__) + " error: Tensor shapes not equal");

    Tensor<T> result(base.m_shape);
    for (size_t i = 0; i < result.m_size; ++i)
        result.m_data[i] = std::pow(base.m_data[i], exp.m_data[i]);

    return result;
}
TMP Tensor<T> Tensor<T>::Pow(const Tensor<T>& base, const T& exp)
{
    Tensor<T> result(base.m_shape);
    for (size_t i = 0; i < result.m_size; ++i)
        result.m_data[i] = std::pow(base.m_data[i], exp);

    return result;
}
TMP Tensor<T> Tensor<T>::Pow(const T& base, const Tensor<T>& exp)
{
    Tensor<T> result(exp.m_shape);
    for (size_t i = 0; i < result.m_size; ++i)
        result.m_data[i] = std::pow(base, exp.m_data[i]);

    return result;
}

TMP Tensor<T> Tensor<T>::Maximum(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.m_shape != b.m_shape)
        throw std::runtime_error(std::string(__func__) + " error: Tensor shapes not equal");
    
    Tensor<T> result(a.m_shape);
    for (size_t i = 0; i < result.m_size; ++i)
        result.m_data[i] = std::max(a.m_data[i], b.m_data[i]);

    return result;
}
TMP Tensor<T> Tensor<T>::Minimum(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.m_shape != b.m_shape)
        throw std::runtime_error(std::string(__func__) + " error: Tensor shapes not equal");
    
    Tensor<T> result(a.m_shape);
    for (size_t i = 0; i < result.m_size; ++i)
        result.m_data[i] = std::min(a.m_data[i], b.m_data[i]);

    return result;
}

TMP Tensor<T> Tensor<T>::Sum(size_t dim, bool keep_dim) const
{
    if (dim >= m_shape.size())
        throw std::runtime_error(std::string("Incorrect dimension in ") + __func__);

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (i == dim) {
            if (keep_dim) {
                new_shape.push_back(1);
            }
        }
        else {
            new_shape.push_back(m_shape[i]);
        }
    }

    std::vector<T> new_data;
    new_data.resize(m_data.size() / m_shape[dim]);

    size_t jump_count = 0;
    size_t jump_size = m_shape[dim] * m_strides[dim];

    size_t sum_cnt = 0;

    for (size_t i = 0; i < new_data.size(); ++i) {
        T result = T(0);
        for (size_t j = 0; j < m_shape[dim]; ++j)
            result += m_data[sum_cnt + j * m_strides[dim] + jump_count * jump_size];
        
        new_data[i] = result;
        sum_cnt += 1;

        if (sum_cnt >= m_strides[dim]) {
            jump_count += 1;
            sum_cnt = 0;
        }
    }
    return Tensor<T>(new_shape, new_data);
}
TMP Tensor<T> Tensor<T>::Mean(size_t dim, bool keep_dim) const
{
    Tensor<T> result = this->Sum(dim, keep_dim);
    T div = static_cast<T>(this->m_shape[dim]);
    
    if (div == T(0))
        throw std::runtime_error("Division by zero");

    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] /= div;
    
    return result;
}
TMP Tensor<T> Tensor<T>::Min(size_t dim, bool keep_dim) const
{
    if (dim >= m_shape.size())
        throw std::runtime_error(std::string("Incorrect dimension in ") + __func__);

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (i == dim) {
            if (keep_dim) {
                new_shape.push_back(1);
            }
        }
        else {
            new_shape.push_back(m_shape[i]);
        }
    }

    std::vector<T> new_data;
    new_data.resize(m_data.size() / m_shape[dim]);

    size_t jump_count = 0;
    size_t jump_size = m_shape[dim] * m_strides[dim];

    size_t sum_cnt = 0;

    for (size_t i = 0; i < new_data.size(); ++i) {
        T min = std::numeric_limits<T>::max();
        for (size_t j = 0; j < m_shape[dim]; ++j) {
            T value = m_data[sum_cnt + j * m_strides[dim] + jump_count * jump_size];
            if (value < min)
                min = value;
        }

        new_data[i] = min;
        sum_cnt += 1;

        if (sum_cnt >= m_strides[dim]) {
            jump_count += 1;
            sum_cnt = 0;
        }
    }
    return Tensor<T>(new_shape, new_data);
}
TMP Tensor<T> Tensor<T>::Max(size_t dim, bool keep_dim) const
{
    if (dim >= m_shape.size())
        throw std::runtime_error(std::string("Incorrect dimension in ") + __func__);

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (i == dim) {
            if (keep_dim) {
                new_shape.push_back(1);
            }
        }
        else {
            new_shape.push_back(m_shape[i]);
        }
    }

    std::vector<T> new_data;
    new_data.resize(m_data.size() / m_shape[dim]);

    size_t jump_count = 0;
    size_t jump_size = m_shape[dim] * m_strides[dim];

    size_t sum_cnt = 0;

    for (size_t i = 0; i < new_data.size(); ++i) {
        T max = std::numeric_limits<T>::min();
        for (size_t j = 0; j < m_shape[dim]; ++j) {
            T value = m_data[sum_cnt + j * m_strides[dim] + jump_count * jump_size];
            if (value > max)
                max = value;
        }

        new_data[i] = max;
        sum_cnt += 1;

        if (sum_cnt >= m_strides[dim]) {
            jump_count += 1;
            sum_cnt = 0;
        }
    }
    return Tensor<T>(new_shape, new_data);
}

TMP Tensor<T> Tensor<T>::Sum(const std::vector<size_t>& dims, bool keep_dim) const
{
    if (dims.size() == 0)
        return *this;

    Tensor<T> result = *this;

    std::vector<size_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<size_t>());

    for (auto& d : sorted_dims)
        result = result.Sum(d, keep_dim);
   
    return result;
}
TMP Tensor<T> Tensor<T>::Mean(const std::vector<size_t>& dims, bool keep_dim) const
{
    if (dims.size() == 0)
        return *this;

    Tensor<T> result = this->Sum(dims, keep_dim);

    T div = T(1);
    for (auto& d : dims)
        div *= static_cast<T>(this->m_shape[d]);

    if (div == T(0))
        throw std::runtime_error("Division by zero");

    for (size_t i = 0; i < result.Size(); ++i)
        result.Data()[i] /= div;

    return result;
}
TMP Tensor<T> Tensor<T>::Min(const std::vector<size_t>& dims, bool keep_dim) const
{
    if (dims.size() == 0)
        return *this;

    Tensor<T> result = *this;

    std::vector<size_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<size_t>());

    for (auto& d : sorted_dims)
        result = result.Min(d, keep_dim);

    return result;
}
TMP Tensor<T> Tensor<T>::Max(const std::vector<size_t>& dims, bool keep_dim) const
{
    if (dims.size() == 0)
        return *this;

    Tensor<T> result = *this;

    std::vector<size_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<size_t>());

    for (auto& d : sorted_dims)
        result = result.Max(d, keep_dim);

    return result;
}

TMP Tensor<size_t> Tensor<T>::Argmin(size_t dim, bool keep_dim) const
{
    if (dim >= m_shape.size())
        throw std::runtime_error(std::string("Incorrect dimension in ") + __func__);

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (i == dim) {
            if (keep_dim) {
                new_shape.push_back(1);
            }
        }
        else {
            new_shape.push_back(m_shape[i]);
        }
    }

    std::vector<size_t> new_data;
    new_data.resize(m_data.size() / m_shape[dim]);

    size_t jump_count = 0;
    size_t jump_size = m_shape[dim] * m_strides[dim];

    size_t sum_cnt = 0;

    for (size_t i = 0; i < new_data.size(); ++i) {
        T min_value = std::numeric_limits<T>::max();
        size_t min_index = 0;
        for (size_t j = 0; j < m_shape[dim]; ++j) {
            T value = m_data[sum_cnt + j * m_strides[dim] + jump_count * jump_size];
            if (value < min_value) {
                min_value = value;
                min_index = j;
            }
        }

        new_data[i] = min_index;
        sum_cnt += 1;

        if (sum_cnt >= m_strides[dim]) {
            jump_count += 1;
            sum_cnt = 0;
        }
    }
    return Tensor<size_t>(new_shape, new_data);
}
TMP Tensor<size_t> Tensor<T>::Argmax(size_t dim, bool keep_dim) const
{
    if (dim >= m_shape.size())
        throw std::runtime_error(std::string("Incorrect dimension in ") + __func__);

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (i == dim) {
            if (keep_dim) {
                new_shape.push_back(1);
            }
        }
        else {
            new_shape.push_back(m_shape[i]);
        }
    }

    std::vector<size_t> new_data;
    new_data.resize(m_data.size() / m_shape[dim]);

    size_t jump_count = 0;
    size_t jump_size = m_shape[dim] * m_strides[dim];

    size_t sum_cnt = 0;

    for (size_t i = 0; i < new_data.size(); ++i) {
        T max_value = std::numeric_limits<T>::min();
        size_t max_index = 0;
        for (size_t j = 0; j < m_shape[dim]; ++j) {
            T value = m_data[sum_cnt + j * m_strides[dim] + jump_count * jump_size];
            if (value > max_value) {
                max_value = value;
                max_index = j;
            }
        }

        new_data[i] = max_index;
        sum_cnt += 1;

        if (sum_cnt >= m_strides[dim]) {
            jump_count += 1;
            sum_cnt = 0;
        }
    }
    return Tensor<size_t>(new_shape, new_data);
}

TMP Tensor<T> Tensor<T>::Matmul(const Tensor<T>& a, const Tensor<T>& b)
{
    size_t dim_a = a.m_shape.size();
    size_t dim_b = b.m_shape.size();

    if (dim_a > 2 || dim_b > 2)
        throw std::runtime_error(std::string(__func__) + " error. Only for 1D/2D tensors");

    if (dim_a == 1 && dim_b == 1)
        return Dot(a, b);

    if (dim_a == 1)
        return Vecmat(a, b);

    if (dim_b == 1)
        return Matvec(a, b);

    size_t a_cols = a.m_shape[1];
    size_t b_rows = b.m_shape[0];

    if (a_cols != b_rows)
        throw std::runtime_error(std::string(__func__) + " error. [A] columns count not equal [B] rows count");

    size_t new_rows = a.m_shape[0];
    size_t new_cols = b.m_shape[1];

    Tensor<T> result = Tensor<T>({ new_rows, new_cols });

    for (size_t r = 0; r < new_rows; ++r) {
        for (size_t c = 0; c < new_cols; ++c) {
            result.At(r, c) = T(0);
            for (size_t i = 0; i < a_cols; ++i) {
                result.At(r, c) += (a.At(r, i) * b.At(i, c));
            }
        }
    }

    return result;
}
TMP Tensor<T> Tensor<T>::Matmul(const Tensor<T>& other) const
{
    return Matmul(*this, other);
}

TMP Tensor<T> Tensor<T>::Vecmat(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.m_shape.size() != 1 || b.m_shape.size() != 2)
        throw std::runtime_error(std::string(__func__) + " error. Tensor dimensions error");

    size_t a_cols = a.m_shape[0];
    size_t b_rows = b.m_shape[0];

    if (a_cols != b_rows)
        throw std::runtime_error(std::string(__func__) + " error. [A] columns count not equal [B] rows count");

    size_t new_shape = b.m_shape[1]; 

    Tensor<T> result = Tensor<T>(std::vector<size_t>{ new_shape });

    for (size_t i = 0; i < new_shape; ++i) {
        result.At(i) = T(0);
        for (size_t c = 0; c < a_cols; ++c) {
            result.At(i) += (a.At(c) * b.At(c, i));
        }
    }

    return result;
}
TMP Tensor<T> Tensor<T>::Matvec(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.m_shape.size() != 2 || b.m_shape.size() != 1)
        throw std::runtime_error(std::string(__func__) + " error. Tensor dimensions error");

    size_t a_cols = a.m_shape[1];
    size_t b_rows = b.m_shape[0];

    if (a_cols != b_rows)
        throw std::runtime_error(std::string(__func__) + " error. [A] columns count not equal [B] rows count");

    size_t new_shape = a.m_shape[0];

    Tensor<T> result = Tensor<T>(std::vector<size_t>{ new_shape });

    for (size_t i = 0; i < new_shape; ++i) {
        result.At(i) = T(0);
        for (size_t c = 0; c < a_cols; ++c) {
            result.At(i) += (a.At(i, c) * b.At(c));
        }
    }

    return result;
}
TMP Tensor<T> Tensor<T>::Dot(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.m_shape.size() != 1 || b.m_shape.size() != 1)
        throw std::runtime_error(std::string(__func__) + " error. Tensor dimensions error");

    size_t a_vec_size = a.m_shape[0];
    size_t b_vec_size = a.m_shape[0];

    if (a_vec_size != b_vec_size)
        throw std::runtime_error(std::string(__func__) + " error. Tensor shapes not equal");

    T result = T(0);
    for (size_t i = 0; i < a_vec_size; ++i) 
        result += a.m_data[i] * b.m_data[i];

    return Tensor<T>(result);
}

TMP Tensor<T> Tensor<T>::Conv1D(const Tensor<T>& input, const Tensor<T>& kernel, const std::vector<size_t>& stride, const std::vector<size_t>& padding)
{
    size_t dim_size = 1;
    if (input.m_shape.size() != dim_size || kernel.m_shape.size() != dim_size || stride.size() != dim_size || padding.size() != dim_size)
        throw std::runtime_error(std::string(__func__) + " error. Check input/kernel/stride/padding dimensions (Must be " + std::to_string(dim_size) + ")");

    std::vector<size_t> out_shape = { 0 };
    for (size_t i = 0; i < out_shape.size(); ++i)
        out_shape[i] = std::floor((input.m_shape[i] - kernel.m_shape[i] + 2 * padding[i]) / stride[i]) + 1;

    Tensor<T> result = Tensor<T>(out_shape, 0);

    auto s = stride[0];
    auto p = padding[0];
    auto K = kernel.m_shape[0];

    for (size_t i = 0; i < out_shape[0]; ++i) {
        T value = T(0);
        for (size_t k = 0; k < K; ++k) {
            int id = s * i + k - p;
            if (id >= 0 && id < input.m_shape[0])
                value += input.At({ (size_t)id }) * kernel.At({k});
        }
    }

    std::vector<size_t> result_shape = { 1, 1 };
    for (size_t i = 0; i < out_shape.size(); ++i)
        result_shape.push_back(out_shape[i]);

    return result.Reshape_Inplace(result_shape);
}
TMP Tensor<T> Tensor<T>::Conv2D(const Tensor<T>& input, const Tensor<T>& kernel, const std::vector<size_t>& stride, const std::vector<size_t>& padding)
{
    size_t dim_size = 2;
    if (input.m_shape.size() != dim_size || kernel.m_shape.size() != dim_size || stride.size() != dim_size || padding.size() != dim_size)
        throw std::runtime_error(std::string(__func__) + " error. Check input/kernel/stride/padding dimensions (Must be " + std::to_string(dim_size) + ")");

    std::vector<size_t> out_shape = { 0, 0 };
    for (size_t i = 0; i < out_shape.size(); ++i)
        out_shape[i] = std::floor((input.m_shape[i] - kernel.m_shape[i] + 2 * padding[i]) / stride[i]) + 1;

    Tensor<T> result = Tensor<T>(out_shape, 0);
   
    std::vector<size_t> result_shape = { 1, 1 };
    for (size_t i = 0; i < out_shape.size(); ++i)
        result_shape.push_back(out_shape[i]);

    return result.Reshape_Inplace(result_shape);
}
TMP Tensor<T> Tensor<T>::Conv3D(const Tensor<T>& input, const Tensor<T>& kernel, const std::vector<size_t>& stride, const std::vector<size_t>& padding)
{
    size_t dim_size = 3;
    if (input.m_shape.size() != dim_size || kernel.m_shape.size() != dim_size || stride.size() != dim_size || padding.size() != dim_size)
        throw std::runtime_error(std::string(__func__) + " error. Check input/kernel/stride/padding dimensions (Must be " + std::to_string(dim_size) + ")");

    std::vector<size_t> out_shape = { 0, 0, 0 };
    for (size_t i = 0; i < out_shape.size(); ++i)
        out_shape[i] = std::floor((input.m_shape[i] - kernel.m_shape[i] + 2 * padding[i]) / stride[i]) + 1;
    
    Tensor<T> result = Tensor<T>(out_shape, 0);

    std::vector<size_t> result_shape = { 1, 1 };
    for (size_t i = 0; i < out_shape.size(); ++i)
        result_shape.push_back(out_shape[i]);

    return result.Reshape_Inplace(result_shape);
}

template class Tensor<bool>;
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<uint8_t>;
template class Tensor<int8_t>;
template class Tensor<uint16_t>;
template class Tensor<int16_t>;
template class Tensor<uint32_t>;
template class Tensor<int32_t>;
template class Tensor<uint64_t>;
template class Tensor<int64_t>;
