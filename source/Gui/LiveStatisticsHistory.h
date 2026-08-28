#pragma once

#include <vector>

#include <EngineInterface/DataPointCollection.h>

class LiveStatisticsHistory
{
public:
    std::vector<DataPointCollection>& getDataRef();
    std::vector<DataPointCollection> const& getDataRef() const;

private:
    std::vector<DataPointCollection> _data;
};
