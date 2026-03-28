#pragma once
#include <vector>
#include <type_traits>
#include <random>

template<typename T>
class RandomGenerator
{
public:
    RandomGenerator()
    {
        gen = std::mt19937(rd());
    }

public:
    T Get(T min, T max)
    {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
            std::uniform_int_distribution distr((int)min, (int)max);
            return distr(gen);
        }
        else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution distr(min, max);
            return distr(gen);
        }
        else if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution distr(min, max);
            return distr(gen);
        }
        else static_assert("Random value must be integral or floating point type");
        return {};
    }
    std::vector<T> Get(T min, T max, size_t count)
    {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
            std::uniform_int_distribution distr((int16_t)min, (int16_t)max);
            std::vector<T> result;
            result.resize(count);
            for (size_t i = 0; i < count; ++i) {
                result[i] = distr(gen);
            }
            return result;
        }
        else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution distr(min, max);
            std::vector<T> result;
            result.resize(count);
            for (size_t i = 0; i < count; ++i) {
                result[i] = distr(gen);
            }
            return result;
        }
        else if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution distr(min, max);
            std::vector<T> result;
            result.resize(count);
            for (size_t i = 0; i < count; ++i)
                result[i] = distr(gen);
            return result;
        }
        else static_assert("Random value must be integral or floating point type");
        return {};
    }


private:
    std::random_device rd;
    std::mt19937 gen;
};
