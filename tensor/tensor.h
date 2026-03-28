#pragma once
#include <vector>

/*
// Для 2D тензоров (матриц)
matmul(a, b)         // Умножение матриц (или dot product для 1D)
dot(a, b)            // Скалярное произведение

// Разложения (опционально)
lu(), qr(), svd()    // LU, QR, SVD разложения
solve(A, b)          // Решение линейной системы
inv()                // Обратная матрица
det()                // Определитель
trace()              // След

// Свертки
conv1d(input, kernel, stride, padding)
conv2d(input, kernel, stride, padding)
conv3d(input, kernel, stride, padding)

// Пулинг
max_pool2d(input, kernel_size, stride)
avg_pool2d(input, kernel_size, stride)
global_avg_pool2d(input)

// Нормализация
batch_norm(input, mean, var, weight, bias)
layer_norm(input)

// Объединение и разделение
cat(tensors, dim)    // Конкатенация вдоль оси
stack(tensors, dim)  // Стек (новая ось)
split(tensor, sizes, dim) // Разделение
chunk(tensor, chunks, dim) // Разделение на равные части

// Расширение
repeat(shape)        // Повторение тензора
expand(shape)        // Расширение без копирования (broadcast)
broadcast_to(shape)  // Бродкастинг

// Копирование
clone()              // Глубокая копия
copy()               // Копирование данных
to(device)           // Перемещение (CPU/GPU)

// Типы
to_type<U>()         // Преобразование типа
type()               // Тип данных
is_contiguous()      // Проверка непрерывности

// Отладка
print()              // Вывод на экран
save(filename)       // Сохранение в файл
load(filename)       // Загрузка из файла

// Если нужны autograd возможности
requires_grad(true)  // Включить вычисление градиентов
backward()           // Обратное распространение
grad()               // Получить градиент
detach()             // Отсоединить от графa вычислений
no_grad()            // Контекст без градиентов
*/
template<typename T>
class Tensor;

template<>
class Tensor<bool>;

template<typename T>
class Tensor
{
public:
    Tensor(const std::vector<size_t>& dimensions);
    Tensor(const std::vector<size_t>& dimensions, const T& value);
    Tensor(const std::vector<size_t>& dimensions, const std::vector<T>& values);
    Tensor(const std::vector<size_t>& dimensions, const std::vector<size_t>& strides, const std::vector<T>& values);
    Tensor(const T& value);

public:
    T& at(size_t i);                       // 1D dimension tensor
    T& at(size_t m, size_t n);             // 2D dimension tensor
    T& at(const std::vector<size_t>& ids); // 3D...ND dimensions tensors

public:
    static Tensor<T> Zeros(const std::vector<size_t>& dimensions);
    static Tensor<T> Ones(const std::vector<size_t>& dimensions);
    static Tensor<T> Full(const std::vector<size_t>& dimensions, const T& value);
    static Tensor<T> Eye(size_t size);
    static Tensor<T> Random(const std::vector<size_t>& dimensions, const T& min, const T& max);
    static Tensor<T> Arange(const T& start, const T& stop, const T& step);
    static Tensor<T> Linspace(const T& start, const T& stop, size_t num);

public:
    auto Shape()   const -> const std::vector<size_t>&;
    auto Strides() const -> const std::vector<size_t>&;
    auto Data()    const -> const std::vector<T>&;
    auto Size()    const -> size_t;

    auto Shape()    -> std::vector<size_t>&;
    auto Strides()  -> std::vector<size_t>&;
    auto Data()     -> std::vector<T>&;

public:
    Tensor<T>& Reshape_Inplace(const std::vector<size_t>& dimensions);
    Tensor<T>& Flatten_Inplace();
    Tensor<T>& Transpose_Inplace();
    Tensor<T>& Transpose_Inplace(size_t dim1, size_t dim2);
    Tensor<T>& Permute_Inplace(const std::vector<size_t>& axis);

    Tensor<T> Reshape(const std::vector<size_t>& dimensions) const;
    Tensor<T> Flatten() const;
    Tensor<T> Transpose() const;
    Tensor<T> Transpose(size_t dim1, size_t dim2) const;
    Tensor<T> Permute(const std::vector<size_t>& axis) const;

public:
    Tensor<T>& Abs_Inplace();
    Tensor<T>& Neg_Inplace();
    Tensor<T>& Exp_Inplace();
    Tensor<T>& Log_Inplace();
    Tensor<T>& Log10_Inplace();
    Tensor<T>& Sqrt_Inplace();
    Tensor<T>& Rsqrt_Inplace();
    Tensor<T>& Sin_Inplace();
    Tensor<T>& Cos_Inplace();
    Tensor<T>& Tan_Inplace();
    Tensor<T>& Floor_Inplace();
    Tensor<T>& Ceil_Inplace();
    Tensor<T>& Round_Inplace();

    Tensor<T> Abs() const;
    Tensor<T> Neg() const;
    Tensor<T> Exp() const;
    Tensor<T> Log() const;
    Tensor<T> Log10() const;
    Tensor<T> Sqrt() const;
    Tensor<T> Rsqrt() const;
    Tensor<T> Sin() const;
    Tensor<T> Cos() const;
    Tensor<T> Tan() const;
    Tensor<T> Floor() const;
    Tensor<T> Ceil() const;
    Tensor<T> Round() const;

public:
    bool All() const;
    bool Any() const;

public:
    Tensor<T>& operator+=(const Tensor<T>& other);
    Tensor<T>& operator-=(const Tensor<T>& other);
    Tensor<T>& operator*=(const Tensor<T>& other);
    Tensor<T>& operator/=(const Tensor<T>& other);

    Tensor<T>& operator+=(const T& val);
    Tensor<T>& operator-=(const T& val);
    Tensor<T>& operator*=(const T& val);
    Tensor<T>& operator/=(const T& val);

public:
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator-(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;
    Tensor<T> operator/(const Tensor<T>& other) const;

    Tensor<T> operator+(const T& val) const;
    Tensor<T> operator-(const T& val) const;
    Tensor<T> operator*(const T& val) const;
    Tensor<T> operator/(const T& val) const;

public:
    Tensor<T>& Add_Inplace(const Tensor<T>& other);
    Tensor<T>& Sub_Inplace(const Tensor<T>& other);
    Tensor<T>& Mul_Inplace(const Tensor<T>& other);
    Tensor<T>& Div_Inplace(const Tensor<T>& other);

    Tensor<T> Add(const Tensor<T>& other) const;
    Tensor<T> Sub(const Tensor<T>& other) const;
    Tensor<T> Mul(const Tensor<T>& other) const;
    Tensor<T> Div(const Tensor<T>& other) const;

public:
    Tensor<bool> operator==(const Tensor<T>& other) const;
    Tensor<bool> operator!=(const Tensor<T>& other) const;
    Tensor<bool> operator> (const Tensor<T>& other) const;
    Tensor<bool> operator< (const Tensor<T>& other) const;
    Tensor<bool> operator>=(const Tensor<T>& other) const;
    Tensor<bool> operator<=(const Tensor<T>& other) const;

public:
    T Sum() const;
    T Prod() const;
    T Mean() const;
    T Var() const;
    T Std() const;
    T Min() const;
    T Max() const;

public:
    Tensor<T> Pow_Inplace(const Tensor<T>& exp);
    Tensor<T> Pow_Inplace(const T& exp);

    static Tensor<T> Pow(const Tensor<T>& base, const Tensor<T>& exp);
    static Tensor<T> Pow(const Tensor<T>& base, const T& exp);
    static Tensor<T> Pow(const T& base, const Tensor<T>& exp);

public:
    static Tensor<T> Maximum(const Tensor<T>& a, const Tensor<T>& b);
    static Tensor<T> Minimum(const Tensor<T>& a, const Tensor<T>& b);

public:
    Tensor<T> Sum(size_t dim, bool keep_dim) const;
    Tensor<T> Mean(size_t dim, bool keep_dim) const;
    Tensor<T> Min(size_t dim, bool keep_dim) const;
    Tensor<T> Max(size_t dim, bool keep_dim) const;

    Tensor<T> Sum(const std::vector<size_t>& dims, bool keep_dim) const;
    Tensor<T> Mean(const std::vector<size_t>& dims, bool keep_dim) const;
    Tensor<T> Min(const std::vector<size_t>& dims, bool keep_dim) const;
    Tensor<T> Max(const std::vector<size_t>& dims, bool keep_dim) const;

    Tensor<size_t> Argmin(size_t dim, bool keep_dim) const;
    Tensor<size_t> Argmax(size_t dim, bool keep_dim) const;
public:
#if 0
    static Tensor<T> Matmul(const Tensor<T>& a, const Tensor<T>& b);  // 2D tensor only
    Tensor<T> Matmul(const Tensor<T>& other) const;                   // 2D tensor only
    Tensor<T>& Matmul_Inplace(const Tensor<T>& other);                // 2D tensor only
#endif
private:
    void CalculateStrides();

private:
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    std::vector<T> m_data;

    size_t m_size;
};

template<>
class Tensor<bool>
{
public:
    Tensor(const std::vector<size_t>& dimensions);
    Tensor(const std::vector<size_t>& dimensions, bool value);
    Tensor(const std::vector<size_t>& dimensions, const std::vector<bool>& values);
    Tensor(const std::vector<size_t>& dimensions, const std::vector<size_t>& strides, const std::vector<bool>& values);

public:
    auto Shape()   const -> const std::vector<size_t>&;
    auto Strides() const -> const std::vector<size_t>&;
    auto Data()    const -> const std::vector<bool>&;
    auto Data() -> std::vector<bool>&;
    auto Size()    const -> size_t;

public:
    bool at(size_t i);                       // 1D dimension tensor
    bool at(size_t m, size_t n);             // 2D dimension tensor
    bool at(const std::vector<size_t>& ids); // 3D...ND dimensions tensors

public:
    Tensor<bool> operator==(const Tensor<bool>& other) const;
    Tensor<bool> operator!=(const Tensor<bool>& other) const;

private:
    void CalculateStrides();

private:
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    std::vector<bool> m_data;

    size_t m_size;
};
