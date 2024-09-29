#pragma once

#include <atomic>
#include <mutex>

#include "Definitions.h"

class PersisterWorker
{
public:
    void runThreadLoop();
    void shutdown();

private:
    std::atomic<bool> _isShutdown{false};
};
