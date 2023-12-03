#pragma once

#include <vector>

#include "DataPointCollection.h"
#include "Definitions.h"

class StatisticsHistory
{
public:
    std::vector<DataPointCollection>& getData();
    std::vector<DataPointCollection> const& getData() const;

private:
    std::vector<DataPointCollection> _data = {DataPointCollection()};
};
