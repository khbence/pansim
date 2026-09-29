#ifdef MATLAB
#include "mex.hpp"
#include "mexAdapter.hpp"
#endif
#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#include <iostream>
#include <memory>
#include <vector>
#include <sstream>

#include "runtime/simulationRuntime.h"
#include "timing.h"

class SimulatorInterface {
private:
  std::vector<unsigned> _stats;
  bool _isInitialized = false;
  std::unique_ptr<runtime::SimulationEngine> engine;

public:
  SimulatorInterface() {
  }

  ~SimulatorInterface() {
    if (engine) {
        engine->finalize();
    }
  }

  void initSimulation(std::string *options, size_t n)
    {
        std::vector<std::string> optVec = std::vector<std::string>{ options, options + n };
        runtime::BootstrapResult bootstrap = runtime::bootstrapSimulation(optVec);
        if (bootstrap.shouldExit) {
            if (!bootstrap.output.empty()) {
                std::cout << bootstrap.output << std::endl;
            }
            _isInitialized = false;
            return;
        }
        engine = std::move(bootstrap.engine);
        _isInitialized = true;
    }

    std::vector<unsigned> runForDay(std::string *options, size_t n)
    {
        if (!_isInitialized) {
            printf("Cannot run uninitialized simulation... Initialize it first.");
            return std::vector<unsigned>();
        }

        std::vector<std::string> optVec = std::vector<std::string>{ options, options + n };

        try {
            BEGIN_PROFILING("runSimulation");
            _stats = engine->runForDay(optVec);

            END_PROFILING("runSimulation");
        } catch (const std::exception& e) {
            std::cerr << e.what();
            return std::vector<unsigned>();
        }
        return _stats;
    }
};

#ifdef MATLAB
class MexFunction : public matlab::mex::Function {
private:
  std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
  // Factory to create MATLAB data arrays
  matlab::data::ArrayFactory factory;

  SimulatorInterface *sim = new SimulatorInterface();

public:
  void operator()(matlab::mex::ArgumentList outputs,
                  matlab::mex::ArgumentList inputs) {
    // Function implementation
    if (inputs.empty()) {
      return;
    }
    if (inputs[0].getType() != matlab::data::ArrayType::CHAR) {
      displayOnMATLAB("The first input must be a char array\n");
      return;
    }
    matlab::data::CharArray charVector1 = inputs[0];
    std::string cmd = charVector1.toAscii();
    displayOnMATLAB("new cmd: " + cmd + "\n");
    if (cmd == "initSimulation") {
      if (inputs[1].getType() != matlab::data::ArrayType::MATLAB_STRING) {
        displayOnMATLAB("The second input must be a string array\n");
        return;
      }
      matlab::data::TypedArray<matlab::data::MATLABString> input = inputs[1];
      std::vector<std::string> m_strValues;
      for (const auto &str : input) {
        m_strValues.push_back(str);
      }
      sim->initSimulation(m_strValues.data(), m_strValues.size());
    } else if (cmd == "runForDay") {
      if (inputs[1].getType() != matlab::data::ArrayType::MATLAB_STRING) {
        displayOnMATLAB("The second input must be a string array\n");
        return;
      }
      matlab::data::TypedArray<matlab::data::MATLABString> input = inputs[1];
      std::vector<std::string> m_strValues;
      for (const auto &str : input) {
        m_strValues.push_back(str);
      }
      auto retvalues = sim->runForDay(m_strValues.data(), m_strValues.size());
      outputs[0] = factory.createArray({1, retvalues.size()}, retvalues.begin(),
                                  retvalues.end());
    } else if (cmd == "delete") {
      delete sim;
    } else {
      displayOnMATLAB("Unknown command: " + cmd + "\n");
    }
  }

  MexFunction() {
    /* mexLock(); */
    displayOnMATLAB("Calling constructor\n");
  }

  virtual ~MexFunction() {
    /* mexUnlock(); //  may be something like a deadlock?? */
    displayOnMATLAB("Calling destructor\n");
  }

  void displayOnMATLAB(const std::stringstream &stream) {
    matlabPtr->feval(
        u"fprintf", 0,
        std::vector<matlab::data::Array>({factory.createScalar(stream.str())}));
  }

  void displayOnMATLAB(const std::string &str) {
    matlabPtr->feval(
        u"fprintf", 0,
        std::vector<matlab::data::Array>({factory.createScalar(str)}));
  }
};
#endif

#ifdef PYTHON
namespace py = pybind11;

PYBIND11_MODULE(pyPanSim, m) {
    py::class_<SimulatorInterface>(m, "SimulatorInterface")
        .def(py::init<>())
        .def("initSimulation", [](SimulatorInterface &self, std::vector<std::string> options) {
            self.initSimulation(options.data(), options.size());
        })
        .def("runForDay", [](SimulatorInterface &self, std::vector<std::string> options) -> std::vector<unsigned int> {
            return self.runForDay(options.data(), options.size());
        });
}
#endif
