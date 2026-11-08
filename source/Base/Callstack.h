#pragma once

#include <string>

#ifdef _WIN32
struct _CONTEXT;
#endif

class Callstack
{
public:
    static std::string capture(int numSkippedFrames = 0);
#ifdef _WIN32
    static std::string capture(_CONTEXT const& context);
#endif
};
