#include "mex.hpp"
#include "mexAdapter.hpp"
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <sstream>

#include "simulation.h"
#include "configTypes.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include <iostream>
#include "agentMeta.h"
#include <inputJSON.h>
#include "timing.h"
#include <cxxopts.hpp>
#include "smallTools.h"
#include "datatypes.h"
#include "version.h"


cxxopts::ParseResult initialize(int argc, char** argv) {
    BEGIN_PROFILING("init");

    auto options = defineProgramParameters();
    config::Simulation_t::addProgramParameters(options);

    //print each arg
    for (int i = 0; i < argc; i++) {
        std::cout << argv[i] << std::endl;
    }

    cxxopts::ParseResult result = options.parse(argc, argv);
    if (result.count("help") != 0) {
        std::cout << options.help() << std::endl;
        exit(EXIT_SUCCESS);
    } else if (result.count("version") != 0) {
        std::cout << config::GIT_VERSION << std::endl;
        exit(EXIT_SUCCESS);
    }

    BEGIN_PROFILING("Device/RNG init");
    RandomGenerator::init(omp_get_max_threads());
    END_PROFILING("Device/RNG init");
    END_PROFILING("init");
    return result;
}

class SimulatorInterface {
private:

  std::vector<std::seed_seq::result_type> _stats;
  cxxopts::ParseResult _initResult;
  bool _isInitialized = false;
  // need to delete this pointer at the very end
  config::Simulation_t *s;

public:
  SimulatorInterface() {
  }

  ~SimulatorInterface() {
    s->finalize();
    delete s;
  }

  void initSimulation(std::string *options, size_t n)
    {
        std::vector<std::string> optVec = std::vector<std::string>{ options, options + n };
        std::vector<char*> csOpts;

        // Boilerplate code for casting vector string to char**
        for (size_t i = 0; i < optVec.size(); i++)
        {
            csOpts.push_back(const_cast<char*>(optVec[i].c_str()));
        }
        
        if (!csOpts.empty())
            _initResult = initialize(csOpts.size(), &csOpts[0]);
        _isInitialized = true;
        s = new config::Simulation_t(_initResult);
    }

    std::vector<std::seed_seq::result_type> runForDay(std::string *options, size_t n)
    {
        if (!_isInitialized) {
            printf("Cannot run uninitialized simulation... Initialize it first.");
            return std::vector<unsigned>();
        }

        // Cast vector string to char **
        std::vector<std::string> optVec = std::vector<std::string>{ options, options + n };
        std::vector<char*> csOpts;

        for (size_t i = 0; i < optVec.size(); i++)
        {
            csOpts.push_back(const_cast<char*>(optVec[i].c_str()));
        }

        try {
            BEGIN_PROFILING("runSimulation");
            // while loop to be on matlab side
            // give char ** and number of args to runForDay
            // get back vector of unsigned
            // convert std::string to char*

            if (!csOpts.empty())
                _stats = s->runForDay(csOpts.size(), &csOpts[0]);

            END_PROFILING("runSimulation");
            // s.finalize();
            // Timing::report();
        } catch (const init::ProgramInit& e) {
            std::cerr << e.what();
            return std::vector<unsigned>();
        }
        return _stats;
    }
};

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
