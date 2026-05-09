#pragma once

#include <EngineInterface/StatisticsRawData.h>

#include "Definitions.h"

class HistogramLiveStatistics
{
public:
    bool isDataAvailable() const;

    std::optional<HistogramData> const& getData() const;
    void reset();
    void update(HistogramData const& data);

private:
    std::optional<HistogramData> _lastHistogramData;
};
