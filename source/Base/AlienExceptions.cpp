#include "AlienExceptions.h"

#include "Callstack.h"

AlienException::AlienException(std::string const& what)
    : std::runtime_error(what)
    , _callstack(Callstack::capture(1))
{}

std::string const& AlienException::getCallstack() const
{
    return _callstack;
}
