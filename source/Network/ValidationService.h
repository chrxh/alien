#pragma once

#include <string>

class ValidationService
{
public:
    static bool isStringValidForDatabase(std::string const& s);
};
