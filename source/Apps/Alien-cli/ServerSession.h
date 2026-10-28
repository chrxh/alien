#pragma once

#include <optional>
#include <string>
#include <thread>

class ServerSession
{
public:
    ServerSession(std::string const& userName, std::string const& password, std::string const& gpuName, bool countAsActive);
    ~ServerSession();

    bool isLoggedIn() const;
    std::optional<std::string> const& getErrorMessage() const;

private:
    bool _loggedIn = false;
    std::optional<std::string> _errorMessage;
    std::jthread _refreshThread;
};
