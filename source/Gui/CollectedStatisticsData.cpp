#include "CollectedStatisticsData.h"

#include <cmath>
#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/RawStatisticsData.h"
#include "EngineInterface/StatisticsConverterService.h"


DataPoint DataPoint::operator+(DataPoint const& other) const
{
    DataPoint result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = values[i] + other.values[i];
    }
    result.summedValues = summedValues + other.summedValues;
    return result;
}

DataPoint DataPoint::operator/(double divisor) const
{
    DataPoint result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = values[i] / divisor;
    }
    result.summedValues = summedValues / divisor;
    return result;
}

DataPointCollection DataPointCollection::operator+(DataPointCollection const& other) const
{
    DataPointCollection result;
    result.time = time + other.time;
    result.numCells = numCells + other.numCells;
    result.numSelfReplicators = numSelfReplicators + other.numSelfReplicators;
    result.numViruses = numViruses + other.numViruses;
    result.numConnections = numConnections + other.numConnections;
    result.numParticles = numParticles + other.numParticles;
    result.averageGenomeCells = averageGenomeCells + other.averageGenomeCells;
    result.totalEnergy = totalEnergy + other.totalEnergy;
    result.numCreatedCells = numCreatedCells + other.numCreatedCells;
    result.numAttacks = numAttacks + other.numAttacks;
    result.numMuscleActivities = numMuscleActivities + other.numMuscleActivities;
    result.numDefenderActivities = numDefenderActivities + other.numDefenderActivities;
    result.numTransmitterActivities = numTransmitterActivities + other.numTransmitterActivities;
    result.numInjectionActivities = numInjectionActivities + other.numInjectionActivities;
    result.numCompletedInjections = numCompletedInjections + other.numCompletedInjections;
    result.numNervePulses = numNervePulses + other.numNervePulses;
    result.numNeuronActivities = numNeuronActivities + other.numNeuronActivities;
    result.numSensorActivities = numSensorActivities + other.numSensorActivities;
    result.numSensorMatches = numSensorMatches + other.numSensorMatches;
    result.numReconnectorCreated = numReconnectorCreated + other.numReconnectorCreated;
    result.numReconnectorRemoved = numReconnectorRemoved + other.numReconnectorRemoved;
    result.numDetonations = numDetonations + other.numDetonations;
    return result;
}

DataPointCollection DataPointCollection::operator/(double divisor) const
{
    DataPointCollection result;
    result.time = time / divisor;
    result.numCells = numCells / divisor;
    result.numSelfReplicators = numSelfReplicators / divisor;
    result.numViruses = numViruses / divisor;
    result.numConnections = numConnections / divisor;
    result.numParticles = numParticles / divisor;
    result.averageGenomeCells = averageGenomeCells / divisor;
    result.totalEnergy = totalEnergy / divisor;
    result.numCreatedCells = numCreatedCells / divisor;
    result.numAttacks = numAttacks / divisor;
    result.numMuscleActivities = numMuscleActivities / divisor;
    result.numDefenderActivities = numDefenderActivities / divisor;
    result.numTransmitterActivities = numTransmitterActivities / divisor;
    result.numInjectionActivities = numInjectionActivities / divisor;
    result.numCompletedInjections = numCompletedInjections / divisor;
    result.numNervePulses = numNervePulses / divisor;
    result.numNeuronActivities = numNeuronActivities / divisor;
    result.numSensorActivities = numSensorActivities / divisor;
    result.numSensorMatches = numSensorMatches / divisor;
    result.numReconnectorCreated = numReconnectorCreated / divisor;
    result.numReconnectorRemoved = numReconnectorRemoved / divisor;
    result.numDetonations = numDetonations / divisor;
    return result;
}

void TimelineLiveStatistics::truncate()
{
    if (!dataPointCollectionHistory.empty() && dataPointCollectionHistory.back().time - dataPointCollectionHistory.front().time > (MaxLiveHistory + 1.0)) {
        dataPointCollectionHistory.erase(dataPointCollectionHistory.begin());
    }
}

void TimelineLiveStatistics::add(TimelineStatistics const& data, uint64_t timestep)
{
    truncate();

    timepoint += ImGui::GetIO().DeltaTime;

    auto newDataPoint = StatisticsConverterService::convert(data, timestep, lastData, lastTimestep);
    newDataPoint.time = timepoint;
    dataPointCollectionHistory.emplace_back(newDataPoint);
    lastData = data;
    lastTimestep = timestep;
}

void TimelineLongtermStatistics::add(TimelineStatistics const& data, uint64_t timestep)
{
    if (!lastData || toDouble(timestep) - dataPointCollectionHistory.back().time > longtermTimestepDelta) {
        auto newDataPoint = StatisticsConverterService::convert(data, timestep, lastData, lastTimestep);
        newDataPoint.time = toDouble(timestep);
        dataPointCollectionHistory.emplace_back(newDataPoint);
        lastData = data;
        lastTimestep = timestep;

        if (dataPointCollectionHistory.size() > 1000) {
            std::vector<DataPointCollection> newDataPoints;
            newDataPoints.reserve(dataPointCollectionHistory.size() / 2);
            for (size_t i = 0; i < (dataPointCollectionHistory.size() - 1) / 2; ++i) {
                DataPointCollection newDataPoint = (dataPointCollectionHistory.at(i * 2) + dataPointCollectionHistory.at(i * 2 + 1)) / 2.0;
                newDataPoint.time = dataPointCollectionHistory.at(i * 2).time;
                newDataPoints.emplace_back(newDataPoint);
            }
            newDataPoints.emplace_back(dataPointCollectionHistory.back());
            dataPointCollectionHistory.swap(newDataPoints);

            longtermTimestepDelta *= 2;
        }
    }
}
