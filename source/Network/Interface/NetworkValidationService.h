#pragma once

#include <string>

#include <Base/Interface/Singleton.h>


class NetworkValidationService
{
    MAKE_SINGLETON(NetworkValidationService);

public:
    bool isStringValidForDatabase(std::string const& s);
};
