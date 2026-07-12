#include "LineageHistory.h"

LineageHistoryData LineageHistory::getCopiedData() const
{
    std::lock_guard lock(_mutex);
    auto copy = _data;
    return copy;
}

std::mutex& LineageHistory::getMutex() const
{
    return _mutex;
}

LineageHistoryData& LineageHistory::getDataRef()
{
    return _data;
}

LineageHistoryData const& LineageHistory::getDataRef() const
{
    return _data;
}
