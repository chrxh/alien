#pragma once

#include <string>

#include "Base/Singleton.h"


class ValidationService
{
    MAKE_SINGLETON(ValidationService);

public:
    bool isStringValidForDatabase(std::string const& s);
};
