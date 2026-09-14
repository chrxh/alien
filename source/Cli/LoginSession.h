#pragma once

#include <optional>
#include <string>
#include <thread>

class LoginSession
{
public:
    LoginSession(std::string const& userName, std::string const& password, std::string const& gpuName);
    ~LoginSession();

    std::optional<std::string> const& getErrorMessage() const;

private:
    std::optional<std::string> _errorMessage;
    std::jthread _refreshThread;
};
