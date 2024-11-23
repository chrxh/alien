#include "StatisticsConverterService.h"

#include <chrono>

#include "Base/Definitions.h"

namespace
{
    template <typename T>
    DataPoint getDataPointBySummation(ColorVector<T> const& values)
    {
        DataPoint result;
        result.summedValues = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.values[i] = toDouble(values[i]);
            result.summedValues += result.values[i];
        }
        return result;
    }

    template <typename T>
    DataPoint getDataPointByMaximation(ColorVector<T> const& values)
    {
        DataPoint result;
        result.summedValues = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.values[i] = toDouble(values[i]);
            result.summedValues = std::max(result.summedValues, result.values[i]);
        }
        return result;
    }

    template<typename T>
    DataPoint getDataPointByAveraging(ColorVector<T> const& summedValue, ColorVector<int> const& numSelfReplicators)
    {
        DataPoint result;
        auto sumSummedValue = 0.0;
        auto sumNumSelfReplicators = 0.0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.values[i] = toDouble(summedValue[i]);
            sumSummedValue += result.values[i];
            sumNumSelfReplicators += numSelfReplicators[i];
            if (numSelfReplicators[i] > 0) {
                result.values[i] /= numSelfReplicators[i];
            }
        }
        result.summedValues = sumNumSelfReplicators > 0 ? sumSummedValue / sumNumSelfReplicators : sumSummedValue;
        return result;
    }

    DataPoint getDataPointForProcessProperty(
        ColorVector<uint64_t> const& values,
        ColorVector<uint64_t> const& lastValues,
        ColorVector<int> const& numCells,
        double deltaTimesteps)
    {
        DataPoint result;
        result.summedValues = 0;
        auto sumNumCells = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (lastValues[i] > values[i] || numCells[i] == 0) {
                result.values[i] = 0;
            } else {
                result.values[i] = toDouble(values[i] - lastValues[i]) / deltaTimesteps / toDouble(numCells[i]);
                result.summedValues += toDouble(values[i] - lastValues[i]) / deltaTimesteps;
                sumNumCells += numCells[i];
            }
        }
        result.summedValues /= toDouble(sumNumCells);
        return result;
    }


}

DataPointCollection StatisticsConverterService::convert(
    TimelineStatistics const& data,
    uint64_t timestep,
    double time,
    std::optional<TimelineStatistics> const& lastData,
    std::optional<uint64_t> lastTimestep)
{
    DataPointCollection result;
    result.time = time;

    auto now = std::chrono::system_clock::now();
    auto unixEpoch = std::chrono::time_point<std::chrono::system_clock>();
    result.systemClock = toDouble(std::chrono::duration_cast<std::chrono::seconds>(now - unixEpoch).count());

    result.numCells = getDataPointBySummation(data.timestep.numCells);
    result.numSelfReplicators = getDataPointBySummation(data.timestep.numSelfReplicators);
    result.numColonies = getDataPointBySummation(data.timestep.numColonies);
    result.numViruses = getDataPointBySummation(data.timestep.numViruses);
    result.numConnections = getDataPointBySummation(data.timestep.numConnections);
    result.numParticles = getDataPointBySummation(data.timestep.numParticles);
    result.averageGenomeCells = getDataPointByAveraging(data.timestep.numGenomeCells, data.timestep.numSelfReplicators);
    result.averageGenomeComplexity = getDataPointByAveraging(data.timestep.genomeComplexity, data.timestep.numSelfReplicators);
    result.varianceGenomeComplexity = getDataPointBySummation(data.timestep.genomeComplexityVariance);
    result.maxGenomeComplexityOfColonies = getDataPointByMaximation(data.timestep.maxGenomeComplexityOfColonies);
    result.totalEnergy = getDataPointBySummation(data.timestep.totalEnergy);

    auto deltaTimesteps = lastTimestep ? toDouble(timestep) - toDouble(*lastTimestep) : 1.0;
    if (deltaTimesteps < NEAR_ZERO) {
        deltaTimesteps = 1.0;
    }

    auto lastDataValue = lastData.value_or(data);
    result.numCreatedCells =
        getDataPointForProcessProperty(data.accumulated.numCreatedCells, lastDataValue.accumulated.numCreatedCells, data.timestep.numCells, deltaTimesteps);
    result.numAttacks =
        getDataPointForProcessProperty(data.accumulated.numAttacks, lastDataValue.accumulated.numAttacks, data.timestep.numCells, deltaTimesteps);
    result.numMuscleActivities = getDataPointForProcessProperty(
        data.accumulated.numMuscleActivities, lastDataValue.accumulated.numMuscleActivities, data.timestep.numCells, deltaTimesteps);
    result.numDefenderActivities = getDataPointForProcessProperty(
        data.accumulated.numDefenderActivities, lastDataValue.accumulated.numDefenderActivities, data.timestep.numCells, deltaTimesteps);
    result.numTransmitterActivities = getDataPointForProcessProperty(
        data.accumulated.numTransmitterActivities, lastDataValue.accumulated.numTransmitterActivities, data.timestep.numCells, deltaTimesteps);
    result.numInjectionActivities = getDataPointForProcessProperty(
        data.accumulated.numInjectionActivities, lastDataValue.accumulated.numInjectionActivities, data.timestep.numCells, deltaTimesteps);
    result.numCompletedInjections = getDataPointForProcessProperty(
        data.accumulated.numCompletedInjections, lastDataValue.accumulated.numCompletedInjections, data.timestep.numCells, deltaTimesteps);
    result.numNervePulses =
        getDataPointForProcessProperty(data.accumulated.numNervePulses, lastDataValue.accumulated.numNervePulses, data.timestep.numCells, deltaTimesteps);
    result.numNeuronActivities = getDataPointForProcessProperty(
        data.accumulated.numNeuronActivities, lastDataValue.accumulated.numNeuronActivities, data.timestep.numCells, deltaTimesteps);
    result.numSensorActivities = getDataPointForProcessProperty(
        data.accumulated.numSensorActivities, lastDataValue.accumulated.numSensorActivities, data.timestep.numCells, deltaTimesteps);
    result.numSensorMatches =
        getDataPointForProcessProperty(data.accumulated.numSensorMatches, lastDataValue.accumulated.numSensorMatches, data.timestep.numCells, deltaTimesteps);
    result.numReconnectorCreated = getDataPointForProcessProperty(
        data.accumulated.numReconnectorCreated, lastDataValue.accumulated.numReconnectorCreated, data.timestep.numCells, deltaTimesteps);
    result.numReconnectorRemoved = getDataPointForProcessProperty(
        data.accumulated.numReconnectorRemoved, lastDataValue.accumulated.numReconnectorRemoved, data.timestep.numCells, deltaTimesteps);
    result.numDetonations =
        getDataPointForProcessProperty(data.accumulated.numDetonations, lastDataValue.accumulated.numDetonations, data.timestep.numCells, deltaTimesteps);

    return result;
}
