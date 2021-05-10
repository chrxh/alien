#pragma once

#include <sstream>

#include "Exceptions.h"

#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define TRY \
    try {

#define CATCH \
    } \
    catch (std::exception const& exception) \
    { \
        std::stringstream ss; \
        ss << exception.what() << std::endl \
           <<__FILENAME__ << ":" << __LINE__;\
        throw BugReportException(ss.str().c_str()); \
    }
