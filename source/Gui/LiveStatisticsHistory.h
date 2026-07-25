#pragma once

#include <vector>

#include <EngineInterface/DataPointCollection.h>

// Short-term history sampled in real time; unlike StatisticsHistory it keeps the raw data points uncompressed
class LiveStatisticsHistory
{
public:
    std::vector<DataPointCollection>& getDataRef();
    std::vector<DataPointCollection> const& getDataRef() const;

private:
    std::vector<DataPointCollection> _data;
};
