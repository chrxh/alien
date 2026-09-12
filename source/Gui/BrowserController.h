#pragma once

#include <chrono>
#include <optional>

#include <Base/Singleton.h>

#include <PersisterInterface/Definitions.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class BrowserController : public MainLoopEntity
{
    MAKE_SINGLETON(BrowserController);

public:
    void process() override;

    void refresh(bool withRetry);
    bool isRefreshing() const;

    BrowserData const& getData() const;

private:
    void init() override;
    void shutdown() override;

    BrowserData _data;
    TaskProcessor _refreshProcessor;
    std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;
};
