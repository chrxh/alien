#pragma once

#include <vector>

#include "DataPointCollection.h"
#include "Definitions.h"

class _StatisticsHistory
{
public:
    std::vector<DataPointCollection>& getData();

private:
    std::vector<DataPointCollection> _data;
};
