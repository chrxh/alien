#pragma once

#include <stdexcept>
#include <string>

class AlienException : public std::runtime_error
{
public:
    explicit AlienException(std::string const& what);

    std::string const& getCallstack() const;

private:
    std::string _callstack;
};

class InitialCheckException : public AlienException
{
public:
    using AlienException::AlienException;
};

class CudaMemoryAllocationException : public AlienException
{
public:
    using AlienException::AlienException;
};
