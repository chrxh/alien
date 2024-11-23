#pragma once

#include <mutex>

class UnlockGuard
{
public:
    UnlockGuard(std::unique_lock<std::mutex>& lock)
        : _lock(lock)
    {
        _lock.unlock();
    }
    ~UnlockGuard() { _lock.lock(); }

private:
    std::unique_lock<std::mutex>& _lock;
};
