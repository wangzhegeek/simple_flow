#ifndef SIMPLEFLOW_UTILS_MATH_UTIL_H
#define SIMPLEFLOW_UTILS_MATH_UTIL_H

#include "types.h"
#include <cmath>
#include <cblas.h>

namespace simpleflow {
namespace utils {

// 数学工具类
class MathUtil {
public:
    // 向量点积
    static Float DotProduct(const Float* x, const Float* y, Int size);
    
    // 向量加法: z = x + y
    static void VectorAdd(const Float* x, const Float* y, Float* z, Int size);
    
    // 向量减法: z = x - y
    static void VectorSubtract(const Float* x, const Float* y, Float* z, Int size);
    
    // 向量缩放: y = alpha * x
    static void VectorScale(const Float* x, Float alpha, Float* y, Int size);
    
    // 向量标量加法: y = x + alpha
    static void VectorAddScalar(const Float* x, Float alpha, Float* y, Int size);
    
    // 向量元素乘积: z = x .* y (element-wise)
    static void VectorMultiply(const Float* x, const Float* y, Float* z, Int size);
    
    // 向量平方: y = x.^2
    static void VectorSquare(const Float* x, Float* y, Int size);
    
    // 向量平方和: sum(x.^2)
    static Float VectorSquareSum(const Float* x, Int size);
    
    // 向量和: sum(x)
    static Float VectorSum(const Float* x, Int size);
    
    // Sigmoid函数: 1 / (1 + exp(-x))
    static Float Sigmoid(Float x);
    
    // 批量Sigmoid函数
    static void Sigmoid(const Float* x, Float* y, Int size);
    
    // 计算对数损失: -y * log(p) - (1 - y) * log(1 - p)
    static Float LogLoss(Float prediction, Float target);
    
    // 随机初始化: 使用Xavier方法
    static void XavierInitialization(Float* weights, Int size, Int fan_in, Int fan_out);
};

} // namespace utils
} // namespace simpleflow

#endif // SIMPLEFLOW_UTILS_MATH_UTIL_H 