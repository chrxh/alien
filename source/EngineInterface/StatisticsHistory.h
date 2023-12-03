#pragma once

#include <mutex>
#include <vector>

#include "DataPointCollection.h"
#include "Definitions.h"

class StatisticsHistory
{
public:
    std::mutex& getMutex() const;
    std::vector<DataPointCollection>& getData();
    std::vector<DataPointCollection> const& getData() const;

private:
    mutable std::mutex _mutex;
    std::vector<DataPointCollection> _data;
};
