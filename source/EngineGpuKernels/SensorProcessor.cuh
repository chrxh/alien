#pragma once

#include "Cell.cuh"
#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    static int constexpr NumScanAngles = 32;
    static int constexpr NumScanPoints = 64;

    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);
    __inline__ __device__ static void searchNeighborhood(SimulationData& data, SimulationResult& result, Cell* cell, Activity& activity);
    __inline__ __device__ static void searchByAngle(SimulationData& data, SimulationResult& result, Cell* cell, Activity& activity);

    __inline__ __device__ static uint8_t convertAngleToData(float angle);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SensorProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Sensor];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__inline__ __device__ void SensorProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    __shared__ Activity activity;
    if (threadIdx.x == 0) {
        int inputExecutionOrderNumber;
        activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);
    }
    __syncthreads();

    if (activity.channels[0] > cudaSimulationParameters.cellFunctionSensorActivityThreshold) {
        switch (cell->cellFunctionData.sensor.mode) {
        case Enums::SensorMode_Neighborhood: {
            searchNeighborhood(data, result, cell, activity);
        } break;
        case Enums::SensorMode_FixedAngle: {
            searchByAngle(data, result, cell, activity);
        } break;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        CellFunctionProcessor::setActivity(cell, activity);
    }
}

__inline__ __device__ void SensorProcessor::searchNeighborhood(SimulationData& data, SimulationResult& result, Cell* cell, Activity& activity)
{
    __shared__ int minDensity;
    __shared__ int maxDensity;
    __shared__ int color;
    __shared__ float refScanAngle;
    __shared__ uint32_t lookupResult;

    if (threadIdx.x == 0) {
        refScanAngle = CellFunctionProcessor::calcLargestGapReferenceAndActualAngle(data, cell, 0).actualAngle;

        minDensity = toInt(cell->cellFunctionData.sensor.minDensity * 255);
        maxDensity = 255;
        color = cell->cellFunctionData.sensor.color;

        lookupResult = 0xffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(NumScanAngles, threadIdx.x, blockDim.x);
    for (float radius = 14.0f; radius <= cudaSimulationParameters.cellFunctionSensorRange; radius += 8.0f) {
        for (int angleIndex = partition.startIndex; angleIndex <= partition.endIndex; ++angleIndex) {
            float angle = 360.0f / NumScanAngles * angleIndex;

            auto delta = Math::unitVectorOfAngle(angle) * radius;
            auto scanPos = cell->absPos + delta;
            data.cellMap.correctPosition(scanPos);
            auto density = static_cast<unsigned char>(data.preprocessedCellFunctionData.densityMap.getDensity(scanPos, color));
            if (density >= minDensity && density <= maxDensity) {
                auto relAngle = Math::subtractAngle(angle, refScanAngle);
                uint32_t angle = convertAngleToData(relAngle);
                uint32_t combined = static_cast<uint32_t>(radius) << 16 | density << 8 | angle;
                atomicMin(&lookupResult, combined);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffff) {
            activity.channels[0] = 1;   //something found
            activity.channels[1] = static_cast<float>((lookupResult >> 8) & 0xff) / 256;  //density
            activity.channels[2] = static_cast<float>(lookupResult >> 16) / 256;  //distance
            activity.channels[3] = static_cast<float>(static_cast<int8_t>(lookupResult & 0xff)) / 256;  //angle
        } else {
            activity.channels[0] = 0;  //nothing found
        }
    }
    __syncthreads();
}

__inline__ __device__ void SensorProcessor::searchByAngle(SimulationData& data, SimulationResult& result, Cell* cell, Activity& activity)
{
    __shared__ int minDensity;
    __shared__ int maxDensity;
    __shared__ int color;
    __shared__ float2 searchDelta;
    __shared__ uint32_t lookupResult;

    if (threadIdx.x == 0) {
        minDensity = toInt(cell->cellFunctionData.sensor.minDensity * 255);
        maxDensity = 255;
        color = cell->cellFunctionData.sensor.color;

        auto refScanAngle = CellFunctionProcessor::calcLargestGapReferenceAndActualAngle(data, cell, 0).actualAngle;
        searchDelta = Math::unitVectorOfAngle(refScanAngle);
        searchDelta = Math::rotateClockwise(searchDelta, cell->cellFunctionData.sensor.angle);

        lookupResult = 0xffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(NumScanPoints, threadIdx.x, blockDim.x);
    for (int distanceIndex = partition.startIndex; distanceIndex <= partition.endIndex; ++distanceIndex) {
        auto distance = 12.0f + cudaSimulationParameters.cellFunctionSensorRange / NumScanPoints * distanceIndex;
        auto scanPos = cell->absPos + searchDelta * distance;
        data.cellMap.correctPosition(scanPos);
        auto density = static_cast<unsigned char>(data.preprocessedCellFunctionData.densityMap.getDensity(scanPos, color));
        if (density >= minDensity && density <= maxDensity) {
            uint32_t combined = static_cast<uint32_t>(distance) << 8 | static_cast<uint32_t>(density);
            atomicMin(&lookupResult, combined);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffff) {
            activity.channels[0] = 1;  //something found
            activity.channels[1] = static_cast<float>(lookupResult & 0xff) / 256;  //density
            activity.channels[2] = static_cast<float>((lookupResult >> 8) & 0xff) / 256;  //distance
        } else {
            activity.channels[0] = 0;  //nothing found
        }
    }
}

__inline__ __device__ uint8_t SensorProcessor::convertAngleToData(float angle)
{
    //0 to 180 degree => 0 to 128
    //-180 to 0 degree => 128 to 256 (= 0)
    angle = remainderf(remainderf(angle, 360.0f) + 360.0f, 360.0f);  //get angle between 0 and 360
    if (angle > 180.0f) {
        angle -= 360.0f;
    }
    int result = static_cast<int>(angle * 128.0f / 180.0f);
    return static_cast<uint8_t>(result);
}
