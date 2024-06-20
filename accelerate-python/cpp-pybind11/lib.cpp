#include <cassert>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;

template <typename T>
py::array_t<T> _matmul(py::array_t<T> a, py::array_t<T> b) {
  // spdlog::debug("a.shape=({},{}), b.shape=({}, {})", a.shape(0), a.shape(1),
  //               b.shape(0), b.shape(1));

  // check inputs
  if (a.ndim() != 2 || b.ndim() != 2) {
    throw std::runtime_error("Number of dimensions of both inputs must be 2");
  }

  if (a.shape(1) != b.shape(0)) {
    throw std::runtime_error("Inner dimensions don't match");
  }

  // get inputs' shapes
  auto n_row = a.shape(0);
  auto n_inner = a.shape(1);
  auto n_col = b.shape(1);

  // construct output tensor
  py::array_t<T> z =
      py::array_t<T>(n_row * n_col); // allocate the buffer for output tensor
  z = z.reshape({n_row, n_col});     // from 1D to 2D

  // get pointer
  auto buf_a = a.request();
  auto buf_b = b.request();
  auto buf_z = z.request();

  T *pa = static_cast<T *>(buf_a.ptr);
  T *pb = static_cast<T *>(buf_b.ptr);
  T *pz = static_cast<T *>(buf_z.ptr);

  size_t ai_base = 0;
  size_t zi = 0;
  for (auto r = 0; r < n_row; ++r) {
    for (auto c = 0; c < n_col; ++c) {
      auto ai = ai_base;
      auto bi = c;
      T v = 0;
      for (auto i = 0; i < n_inner; ++i) {
        v += pa[ai] * pb[bi];
        bi += n_col;
        ai += 1;
      }
      pz[zi] = v;
      zi += 1;
    }
    ai_base += n_inner;
  }

  return z;
}

py::array_t<int32_t> matmul_int32(py::array_t<int32_t> a,
                                  py::array_t<int32_t> b) {
  return _matmul<int32_t>(a, b);
}

py::array_t<float> matmul_float32(py::array_t<float> &a,
                                  py::array_t<float> &b) {
  return _matmul<float>(a, b);
}

// wrap as Python module
PYBIND11_MODULE(lib, m) {
  // setup logger
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::info);

  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "/tmp/prototypes/cpp_pybind11_lib.log", true);
  file_sink->set_level(spdlog::level::trace);

  auto logger = std::make_shared<spdlog::logger>(
      "lib_cpp", spdlog::sinks_init_list({console_sink, file_sink}));
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::trace);

  m.doc() = "matmul implementation in C++";
  m.def("matmul_int32", &matmul_int32,
        "matrix multiplication with int32 data type");
  m.def("matmul_float32", &matmul_float32,
        "matrix multiplication with float32 data type");
}