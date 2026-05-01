#include <string>
#include <iostream>
#include "tensor.h"

void Test();

template<typename T>
void Check(const Tensor<T>& result, const Tensor<T>& out, int line)
{
    bool shape_res = (result.Shape() == out.Shape());
    if (!shape_res)
        std::cout << "Incorrect shapes " << line << std::endl;

    bool srtides_res = (result.Strides() == out.Strides());
    if (!srtides_res)
        std::cout << "Incorrect strides" << line << std::endl;

    bool data_res = (result.Data() == out.Data());
    if (!srtides_res)
        std::cout << "Incorrect data." << line << std::endl;
    
    bool all_res = shape_res && srtides_res && data_res;
    std::cout << (all_res ? "OK " : "FAIL ") << line << std::endl;
}

template<>
void Check<float>(const Tensor<float>& result, const Tensor<float>& out, int line)
{
    bool shape_res = (result.Shape() == out.Shape());
    if (!shape_res)
        std::cout << "Incorrect shapes " << line << std::endl;

    bool srtides_res = (result.Strides() == out.Strides());
    if (!srtides_res)
        std::cout << "Incorrect strides " << line << std::endl;

    bool data_size_res = (result.Size() == out.Size());
    if (!data_size_res)
        std::cout << "Incorrect data size " << line << std::endl;
    
    bool data_res = true;
    for (size_t i = 0; i < result.Size(); ++i) {
        float result_val = result.Data()[i];
        float out_val = out.Data()[i];
        float epsilon = 0.001;
        if (std::abs(result_val - out_val) >= epsilon) {
            std::cout << "Incorrect data " << line << std::endl;
            data_res = false;
        }
    }

    bool all_res = shape_res && srtides_res && data_size_res && data_res;
    std::cout << (all_res ? "OK " : "FAIL ") << line << std::endl;
}

int main()
{
    Test();
    char c;
    std::cin >> c;
    return 0;
}

void Test()
{
    {
        Tensor<int32_t> t = Tensor<int32_t>::Arange(2, 10, 2);
        std::cout << (t.Data() == std::vector<int32_t>{ 2, 4, 6, 8 } ? "OK " : "FAIL ") << __LINE__ << std::endl;
    }
    {
        Tensor<int32_t> t({ 5 }, { 10, 20, 30, 40, 50 });
        std::cout << (t.At(2) == 30 ? "OK " : "FAIL ") << __LINE__ << std::endl;
    }
    {
        Tensor<int32_t> t({ 3, 4 }, { 1,2,3,4,5,6,7,8,9,10,11,12 });
        std::cout << (t.At(1, 2) == 7 ? "OK " : "FAIL ") << __LINE__ << std::endl;
    }
    {
        Tensor<int32_t> t({ 2, 3, 4 }, { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 });
        std::cout << (t.At({ 1, 2, 3 }) == 23 ? "OK " : "FAIL ") << __LINE__ << std::endl;
    }
    {
        Tensor<int32_t> in({ 2, 4 }, { 1,2,3,4,5,6,7,8 });
        Tensor<int32_t> out({ 4, 2 }, { 1,2,3,4,5,6,7,8 });
        Tensor<int32_t> result = in.Reshape({ 4,2 });
        in.Reshape_Inplace({ 4,2 });

        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3, 4 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> out({ 24 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> result = in.Flatten();
        in.Flatten_Inplace();

        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3, 4 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> out({ 3, 2, 4 }, {4, 12, 1}, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> result = in.Transpose(0, 1);
        in.Transpose_Inplace(0, 1);

        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<int32_t> in ({ 2, 3, 4 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> out({ 4, 2, 3 }, {1, 12, 4}, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> result = in.Permute({ 2, 0, 1 });
        in.Permute_Inplace({ 2, 0, 1 });

        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<int32_t> in ({ 5 }, { -2, 0, 3, -5, 7 });
        Tensor<int32_t> out({ 5 }, {  2, 0, 3,  5, 7 });
        Tensor<int32_t> result = in.Abs();
        in.Abs_Inplace();

        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<int32_t> in ({ 5 }, { -2, 0,  3, -5,  7 });
        Tensor<int32_t> out({ 5 }, { 2,  0, -3,  5, -7 });
        Tensor<int32_t> result = in.Neg();
        in.Neg_Inplace();

        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<float> in ({ 5 }, { 0, 1, 2, -1, -2 });
        Tensor<float> out({ 5 }, { 1, 2.718, 7.389, 0.367, 0.135 });
        Tensor<float> result = in.Exp();
        in.Exp_Inplace();
       
        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<float> in({ 5 }, { 1,2,3,4,5 });
        Tensor<float> out({ 5 }, { 0, 0.693, 1.098, 1.386, 1.609 });
        Tensor<float> result = in.Log();
        in.Log_Inplace();
      
        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<float> in({ 5 }, { 1, 10, 100, 1000, 10000  });
        Tensor<float> out({ 5 }, { 0, 1, 2, 3, 4 });
        Tensor<float> result = in.Log10();
        in.Log10_Inplace();
      
        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<float> in({ 5 }, { 0, 1, 4, 9, 16 });
        Tensor<float> out({ 5 }, { 0, 1, 2, 3, 4 });
        Tensor<float> result = in.Sqrt();
        in.Sqrt_Inplace();
      
        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<float> in({ 5 }, { 1, 4, 9, 16, 25 });
        Tensor<float> out({ 5 }, { 1, 0.5, 0.333, 0.25, 0.2 });
        Tensor<float> result = in.Rsqrt();
        in.Rsqrt_Inplace();
       
        Check(result, out, __LINE__);
        Check(in, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        Tensor<int32_t> out({ 2 }, {1}, { 6, 15 });
        Tensor<int32_t> result = in.Sum(1, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        Tensor<int32_t> out({ 2 }, { 1 }, { 2, 5 });
        Tensor<int32_t> result = in.Mean(1, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3, 4 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> out({ 4 }, { 1 }, { 60, 66, 72, 78 });
        Tensor<int32_t> result = in.Sum({ 0, 1 }, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3, 4 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> out({ 4 }, { 1 }, { 10, 11, 12, 13 });
        Tensor<int32_t> result = in.Mean({ 0, 1 }, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3 }, { 1, 5, 3, 8, 2, 4 });
        Tensor<int32_t> out({ 2 }, { 1 }, { 1, 2 });
        Tensor<int32_t> result = in.Min(1, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3 }, { 1, 5, 3, 8, 2, 4 });
        Tensor<int32_t> out({ 2 }, { 1 }, { 5, 8 });
        Tensor<int32_t> result = in.Max(1, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3, 4 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> out({ 4 }, { 1 }, { 0, 1, 2, 3 });
        Tensor<int32_t> result = in.Min({0,1}, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3, 4 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
        Tensor<int32_t> out({ 4 }, { 1 }, { 20, 21, 22, 23 });
        Tensor<int32_t> result = in.Max({ 0,1 }, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3 }, { 1, 5, 3, 8, 2, 4 });
        Tensor<size_t> out({ 2 }, { 1 }, { 0, 1 });
        Tensor<size_t> result = in.Argmin(1, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> in({ 2, 3 }, { 1, 5, 3, 8, 2, 4 });
        Tensor<size_t> out({ 2 }, { 1 }, { 1, 0 });
        Tensor<size_t> result = in.Argmax(1, false);

        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> a({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        Tensor<int32_t> b({ 3, 2 }, { 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 2, 2 }, { 2, 1 }, { 58, 64, 139, 154 });
        Tensor<int32_t> result = Tensor<int32_t>::Matmul(a, b);
       
        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> a({ 2, 2 }, { 1, 2, 3, 4 });
        Tensor<int32_t> b({ 2, 2 }, { 5, 6, 7, 8 });
        Tensor<int32_t> out({ 2, 2 }, { 2, 1 }, { 19, 22, 43, 50 });
        Tensor<int32_t> result = Tensor<int32_t>::Matmul(a, b);
  
        Check(result, out, __LINE__);
    }
    {
        Tensor<int32_t> a({ 1, 3 }, { 1, 2, 3 });
        Tensor<int32_t> b({ 3, 1 }, { 4, 5, 6 });
        Tensor<int32_t> out({ 1, 1 }, { 1, 1 }, { 32 });
        Tensor<int32_t> result = Tensor<int32_t>::Matmul(a, b);

        Check(result, out, __LINE__);
    }
    {
        // Matvec
        Tensor<int32_t> a({ 4, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> b({ 3 }, { 2, 3, 4 });
        Tensor<int32_t> out({ 4 }, { 1 }, { 20, 47, 74, 101 });
        Tensor<int32_t> result = Tensor<int32_t>::Matmul(a, b);
   
        Check(result, out, __LINE__);
    }
    {
        // Matvec
        Tensor<int32_t> a({ 2, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8 });
        Tensor<int32_t> b({ 4 }, { 1, 0, 2, 0 });
        Tensor<int32_t> out({ 2 }, { 1 }, { 7, 19 });
        Tensor<int32_t> result = Tensor<int32_t>::Matmul(a, b);

        Check(result, out, __LINE__);
    }
    {
        // Vecmat
        Tensor<int32_t> a({ 3 }, { 1, 2, 3 });
        Tensor<int32_t> b({ 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 4 }, { 1 }, { 38, 44, 50, 56 });
        Tensor<int32_t> result = Tensor<int32_t>::Matmul(a, b);

        Check(result, out, __LINE__);
    }
    {
        // Vecmat
        Tensor<int32_t> a({ 4 }, { 1, 0, 2, 0 });
        Tensor<int32_t> b({ 4, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8 });
        Tensor<int32_t> out({ 2 }, { 1 }, { 11, 14 });
        Tensor<int32_t> result = Tensor<int32_t>::Matmul(a, b);

        Check(result, out, __LINE__);
    }
    {
        // Unsqueeze 0 1D
        Tensor<int32_t> a({ 4 }, { 1, 2, 3, 4 });
        Tensor<int32_t> out({ 1, 4 }, {4, 1}, { 1, 2, 3, 4 });
        Tensor<int32_t> result = a.Unsqueeze(0);
   
        Check(result, out, __LINE__);
    }
    {
        // Unsqueeze 1 1D
        Tensor<int32_t> a({ 4 }, { 1, 2, 3, 4 });
        Tensor<int32_t> out({ 4, 1 }, { 1, 1 }, { 1, 2, 3, 4 });
        Tensor<int32_t> result = a.Unsqueeze(1);
  
        Check(result, out, __LINE__);
    }
    {
        // Unsqueeze 1 2D
        Tensor<int32_t> a({ 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 3, 1, 4 }, { 4, 4, 1 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> result = a.Unsqueeze(1);

        Check(result, out, __LINE__);
    }
    {
        // Unsqueeze 0 2D
        Tensor<int32_t> a({ 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 1, 3, 4 }, { 12, 4, 1 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> result = a.Unsqueeze(0);

        Check(result, out, __LINE__);
    }
    {
        // Squeeze 0 
        Tensor<int32_t> a({ 1, 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 3, 4 }, { 4, 1 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> result = a.Squeeze(0);

        Check(result, out, __LINE__);
    }
    {
        // Squeeze 1
        Tensor<int32_t> a({ 3, 1, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 3, 4 }, { 4, 1 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> result = a.Squeeze(1);

        Check(result, out, __LINE__);
    }
    {
        // Squeeze 2
        Tensor<int32_t> a({ 3, 4, 1 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 3, 4 }, { 4, 1 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> result = a.Squeeze(2);

        Check(result, out, __LINE__);
    }
    {
        // Squeeze 2
        Tensor<int32_t> a({ 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> out({ 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        Tensor<int32_t> result = a.Squeeze(0);

        Check(result, out, __LINE__);
    }
    {
        // TEMP: check conv result shape sizes

        auto t1 = Tensor<int32_t>::Conv1D({ {100},        1 }, { { 5 }, 2 },       { 2 },       { 1 }      ); //  [49]
        auto t2 = Tensor<int32_t>::Conv1D({ {100},        1 }, { { 5 }, 2 },       { 3 },       { 0 }      ); //  [32]
                                                                
        auto t3 = Tensor<int32_t>::Conv2D({ {32, 32},     1 }, { { 3, 3 }, 0 },    { 2, 2 },    { 1, 1 }   ); //  [16, 16]
        auto t4 = Tensor<int32_t>::Conv2D({ {64, 48},     1 }, { { 5, 5 }, 0 },    { 3, 2 },    { 1, 1 }   ); //  [21, 23]
                                                                
        auto t5 = Tensor<int32_t>::Conv3D({ {16, 32, 32}, 1 }, { { 3, 3, 3 }, 0 }, { 2, 2, 2 }, { 1, 1, 1 }); //  [8, 16, 16]
        auto t6 = Tensor<int32_t>::Conv3D({ {32, 64, 64}, 1 }, { { 5, 3, 3 }, 0 }, { 3, 2, 2 }, { 2, 1, 1 }); //  [11, 32, 32]
    }
}

