#include "smallTools.h"

std::string separator() {
#ifdef _WIN32
    return std::string("\\");
#else
    return std::string("/");
#endif
}

std::vector<double> splitStringDouble(std::string probsString, char sep) {
    std::stringstream ss(probsString);
    std::string arg;
    std::vector<double> params;
    for (char i; ss >> i;) {
        arg.push_back(i);
        if (ss.peek() == sep) {
            if (arg.length() > 0 && isdigit(arg[0])) {
                params.push_back(atof(arg.c_str()));
                arg.clear();
            }
            ss.ignore();
        }
    }
    if (arg.length() > 0 && isdigit(arg[0])) {
        params.push_back(atof(arg.c_str()));
        arg.clear();
    }
    return params;
}


std::vector<float> splitStringFloat(std::string probsString, char sep) {
    std::stringstream ss(probsString);
    std::string arg;
    std::vector<float> params;
    for (char i; ss >> i;) {
        arg.push_back(i);
        if (ss.peek() == sep) {
            if (arg.length() > 0 && isdigit(arg[0])) {
                params.push_back(atof(arg.c_str()));
                arg.clear();
            }
            ss.ignore();
        }
    }
    if (arg.length() > 0 && isdigit(arg[0])) {
        params.push_back(atof(arg.c_str()));
        arg.clear();
    }
    return params;
}

std::vector<int> splitStringInt(std::string probsString, char sep) {
    std::stringstream ss(probsString);
    std::string arg;
    std::vector<int> params;
    for (char i; ss >> i;) {
        arg.push_back(i);
        if (ss.peek() == sep) {
            if (arg.length() > 0 && isdigit(arg[0])) {
                params.push_back(atoi(arg.c_str()));
                arg.clear();
            }
            ss.ignore();
        }
    }
    if (arg.length() > 0 && isdigit(arg[0])) {
        params.push_back(atoi(arg.c_str()));
        arg.clear();
    }
    return params;
}

std::vector<std::string> splitStringString(std::string header, char sep) {
    std::stringstream ss(header);
    std::string arg;
    std::vector<std::string> params;
    for (char i; ss >> i;) {
        arg.push_back(i);
        if (ss.peek() == sep) {
            if (arg.length() > 0) {
                params.push_back(arg);
                arg.clear();
            }
            ss.ignore();
        }
    }
    if (arg.length() > 0) {
        params.push_back(arg);
        arg.clear();
    }
    return params;
}