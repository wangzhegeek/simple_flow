#include "simpleflow/utils/math_util.h"
#include <cmath>
#include <cblas.h>
#include <algorithm>

namespace simpleflow {
namespace utils {

Float MathUtil::DotProduct(const Float* x, const Float* y, Int size) {
    return cblas_sdot(size, x, 1, y, 1);
}

void MathUtil::VectorAdd(const Float* x, const Float* y, Float* z, Int size) {
    std::copy(x, x + size, z);
    cblas_saxpy(size, 1.0, y, 1, z, 1);
}

void MathUtil::VectorSubtract(const Float* x, const Float* y, Float* z, Int size) {
    std::copy(x, x + size, z);
    cblas_saxpy(size, -1.0, y, 1, z, 1);
}

void MathUtil::VectorScale(const Float* x, Float alpha, Float* y, Int size) {
    std::copy(x, x + size, y);
    cblas_sscal(size, alpha, y, 1);
}

void MathUtil::VectorAddScalar(const Float* x, Float alpha, Float* y, Int size) {
    for (Int i = 0; i < size; ++i) {
        y[i] = x[i] + alpha;
    }
}

void MathUtil::VectorMultiply(const Float* x, const Float* y, Float* z, Int size) {
    for (Int i = 0; i < size; ++i) {
        z[i] = x[i] * y[i];
    }
}

void MathUtil::VectorSquare(const Float* x, Float* y, Int size) {
    for (Int i = 0; i < size; ++i) {
        y[i] = x[i] * x[i];
    }
}

Float MathUtil::VectorSquareSum(const Float* x, Int size) {
    Float sum = 0.0;
    for (Int i = 0; i < size; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

Float MathUtil::VectorSum(const Float* x, Int size) {
    Float sum = 0.0;
    for (Int i = 0; i < size; ++i) {
        sum += x[i];
    }
    return sum;
}

Float MathUtil::Sigmoid(Float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void MathUtil::Sigmoid(const Float* x, Float* y, Int size) {
    for (Int i = 0; i < size; ++i) {
        y[i] = Sigmoid(x[i]);
    }
}

Float MathUtil::LogLoss(Float prediction, Float target) {
    Float p = std::max(std::min(prediction, 1.0f - 1e-7f), 1e-7f);
    return -target * std::log(p) - (1.0f - target) * std::log(1.0f - p);
}

void MathUtil::XavierInitialization(Float* weights, Int size, Int fan_in, Int fan_out) {
    Float scale = std::sqrt(6.0 / (fan_in + fan_out));
    for (Int i = 0; i < size; ++i) {
        weights[i] = scale * (2.0 * static_cast<Float>(std::rand()) / RAND_MAX - 1.0);
    }
}

} // namespace utils
} // namespace simpleflow 