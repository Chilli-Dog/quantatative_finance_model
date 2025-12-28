#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <numeric>

namespace py = pybind11;

// A simple C++ SMA calculation
std::vector<double> calculate_sma_cpp(std::vector<double> prices, size_t window) {
    std::vector<double> result;
    result.reserve(prices.size());

    for (size_t i = 0; i < prices.size(); ++i) {
        if (i < window - 1) {
            result.push_back(0.0);
        } else {
            double sum = std::accumulate(prices.begin() + i - window + 1, prices.begin() + i + 1, 0.0);
            result.push_back(sum / (double)window);
        }
    }
    return result;
}

// Wrap the function into a module named 'cpp_trading'
PYBIND11_MODULE(cpp_trading, m) {
    m.def("calculate_sma", &calculate_sma_cpp, "Calculate SMA using C++");
}