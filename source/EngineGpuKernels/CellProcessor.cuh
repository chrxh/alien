#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "Base.cuh"
#include "ConstructorHelper.cuh"
#include "Map.cuh"
#include "MuscleProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "ParameterCalculator.cuh"
#include "TOs.cuh"

class CellProcessor
{
public:
    __inline__ __device__ static void collectCellTypeOperations(SimulationData& data);
    __inline__ __device__ static bool isCellReady(SimulationData& data, Object* object);

    __inline__ __device__ static void aging(SimulationData& data);
    __inline__ __device__ static void cellStateTransition_calcFutureState(SimulationData& data);
    __inline__ __device__ static void cellStateTransition_applyNextState(SimulationData& data);
    __inline__ __device__ static void headUpdate_calcFutureValue(SimulationData& data);
    __inline__ __device__ static void headUpdate_applyFutureValue(SimulationData& data);

    __inline__ __device__ static void updateCellEvents(SimulationData& data);

    __inline__ __device__ static void performEnergyFlow(SimulationData& data);

    __inline__ __device__ static void decay(SimulationData& data);

private:
    __inline__ __device__ static float getInitialAngelSpan(Object* cell, int connectionIndex1, int connectionIndex2);
    __inline__ __device__ static float getInitialAngelSpan(Object* cell, Object* connectedObject1, Object* connectedObject2);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellProcessor::collectCellTypeOperations(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        if (object->type == ObjectType_Cell && object->typeData.cell.cellType != CellType_Base) {
            if (object->typeData.cell.cellType == CellType_Detonator && object->typeData.cell.cellTypeData.detonator.state == DetonatorState_Activated) {
                data.cellTypeOperations[object->typeData.cell.cellType].tryAddEntry(CellTypeOperation{object});
            } else if (isCellReady(data, object)) {
                data.cellTypeOperations[object->typeData.cell.cellType].tryAddEntry(CellTypeOperation{object});
            }
        }
    }
}

__inline__ __device__ bool CellProcessor::isCellReady(SimulationData& data, Object* object)
{
    return object->typeData.cell.cellState != CellState_Constructing && object->typeData.cell.cellState != CellState_Activating
        && object->typeData.cell.activationTime == 0;
}

__inline__ __device__ void CellProcessor::aging(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = data.entities.objects.at(index);
        if (object->fixed || object->type == ObjectType_Solid || object->type == ObjectType_Fluid) {
            continue;
        }
        uint32_t* age = nullptr;
        if (object->type == ObjectType_Cell) {
            age = &object->typeData.cell.age;
        } else if (object->type == ObjectType_FreeCell) {
            age = &object->typeData.freeCell.age;
        }
        ++*age;


        if (cudaSimulationParameters.colorTransitionRulesToggle.value) {
            int transitionDuration;
            int targetColor;
            auto color = calcMod(object->color, MAX_COLORS);
            auto index = ParameterCalculator::getFirstMatchingLayerOrBase(data, object->pos, cudaSimulationParameters.colorTransitionRules);
            if (index == -1) {
                transitionDuration = cudaSimulationParameters.colorTransitionRules.baseValue[color].duration;
                targetColor = cudaSimulationParameters.colorTransitionRules.baseValue[color].targetColor;
            } else {
                transitionDuration = cudaSimulationParameters.colorTransitionRules.layerValues[index].value[color].duration;
                targetColor = cudaSimulationParameters.colorTransitionRules.layerValues[index].value[color].targetColor;
            }
            if (transitionDuration > 0 && *age > transitionDuration) {
                object->color = targetColor;
                *age = 0;
            }
        }
        if (object->type == ObjectType_Cell) {
            if (object->typeData.cell.cellState == CellState_Ready && object->typeData.cell.activationTime > 0) {
                --object->typeData.cell.activationTime;
            }
        }
    }
}


__inline__ __device__ void CellProcessor::cellStateTransition_calcFutureState(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }

        bool isNeighborActivating = false;
        if (object->numConnections > 0) {
            isNeighborActivating = object->connections[0].object->typeData.cell.cellState == CellState_Activating;
        }

        auto origCellState = object->typeData.cell.cellState;
        auto cellState = origCellState;

        if (object->fixed) {
            cellState = CellState_Ready;
        } else {
            if (origCellState == CellState_Activating) {
                cellState = CellState_Ready;
                if (cudaSimulationParameters.cellAgeLimiterToggle.value && cudaSimulationParameters.resetCellAgeAfterActivation.value) {
                    atomicExch(&object->typeData.cell.age, 0);
                }
            } else if (origCellState == CellState_Constructing) {
                if (isNeighborActivating) {
                    cellState = CellState_Activating;
                }
            }
            if (object->type != ObjectType_Cell) {
                cellState = origCellState;
            }
            if (object->typeData.cell.lastUpdate > 2 * Cell::UpdateInterval + 1) {
                cellState = CellState_Dying;
            }
        }
        object->tempValue1.as_uint32_float.uint32Part = cellState;
    }
}

__inline__ __device__ void CellProcessor::cellStateTransition_applyNextState(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        object->typeData.cell.cellState = object->tempValue1.as_uint32_float.uint32Part;
        object->tempValue1.as_uint32_float.uint32Part = 0;
    }
}

__inline__ __device__ void CellProcessor::headUpdate_calcFutureValue(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if (*data.timestep % Cell::UpdateInterval == 0) {
            object->typeData.cell.creature->creatureIndex = VALUE_NOT_SET_UINT64;
        }

        auto update = false;
        if (object->typeData.cell.headUpdateId != object->typeData.cell.creature->headUpdateId) {
            if (!object->typeData.cell.headCell) {
                for (int i = 0, j = object->numConnections; i < j; ++i) {
                    auto const& otherObject = object->connections[i].object;
                    if (otherObject->type != ObjectType_Cell) {
                        continue;
                    }
                    if (!object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
                        continue;
                    }
                    if (otherObject->typeData.cell.headUpdateId > object->typeData.cell.headUpdateId) {
                        auto frontAngle_otherObject_object = Math::getNormalizedAngle(
                            -getInitialAngelSpan(otherObject, object, otherObject->connections[0].object) - otherObject->typeData.cell.frontAngle, -180.0f);

                        auto frontAngle_object_otherObject = Math::getNormalizedAngle(180.0f - frontAngle_otherObject_object, -180.0f);
                        auto frontAngle_object_connection0 =
                            Math::getNormalizedAngle(frontAngle_object_otherObject + getInitialAngelSpan(object, 0, i), -180.0f);
                        object->tempValue2.as_uint32_float.uint32Part = otherObject->typeData.cell.headUpdateId;
                        object->tempValue2.as_uint32_float.floatPart = frontAngle_object_connection0;
                        update = true; 
                        break;
                    }
                }
            }
        }
        if (!update) {
            object->tempValue2.as_uint32_float.uint32Part = object->typeData.cell.headUpdateId;
        }
    }
}

__inline__ __device__ void CellProcessor::headUpdate_applyFutureValue(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto const& creature = object->typeData.cell.creature;
        if (*data.timestep % Cell::UpdateInterval == 0) {
            if (alienAtomicExch64(&creature->creatureIndex, static_cast<uint64_t>(0)) == VALUE_NOT_SET_UINT64) {
                ++creature->headUpdateId;
            }
        }

        if (!object->typeData.cell.headCell) {
            ++object->typeData.cell.lastUpdate;
        }

        if (object->typeData.cell.headCell) {
            if (*data.timestep % Cell::UpdateInterval == 1) {
                object->typeData.cell.headUpdateId = creature->headUpdateId;
                object->typeData.cell.frontAngle = creature->genome->frontAngle;
            }
        } else {
            auto const& newHeadUpdateId = object->tempValue2.as_uint32_float.uint32Part;
            auto const& newFrontAngle = object->tempValue2.as_uint32_float.floatPart;

            if (newHeadUpdateId > object->typeData.cell.headUpdateId) {
                object->typeData.cell.headUpdateId = newHeadUpdateId;
                object->typeData.cell.frontAngle = newFrontAngle;
                object->typeData.cell.lastUpdate = 0;
            }

            object->tempValue2.as_uint64 = 0;
        }
    }
}

__inline__ __device__ void CellProcessor::updateCellEvents(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if (object->typeData.cell.eventCounter > 0) {
            --object->typeData.cell.eventCounter;
        }
    }
}

__inline__ __device__ void CellProcessor::performEnergyFlow(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if (object->numConnections == 0) {
            continue;
        }
        //if (object->typeData.cell.cellState == CellState_Constructing || object->typeData.cell.cellState == CellState_Activating) {
        //    continue;
        //}
        auto i = *data.timestep % object->numConnections;
        auto& connectedObject = object->connections[i].object;
        // Skip if connected object is not a Cell
        if (connectedObject->type != ObjectType_Cell) {
            continue;
        }

        // Flow of usable energy
        {
            auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);
            auto cellNormalEnergy = cudaSimulationParameters.normalCellEnergy.value[object->color];

            auto needsEnergy = [](Object* obj) {
                return (obj->typeData.cell.cellState == CellState_Ready)
                    && obj->typeData.cell.constructorAvailable && obj->typeData.cell.creature
                    && !ConstructorHelper::isFinished(obj->typeData.cell.constructor, *obj->typeData.cell.creature->genome);
            };
            auto lowEnergy = [&](Object* obj) { return obj->typeData.cell.usableEnergy < cellNormalEnergy; };

            auto cellNeedsEnergy = needsEnergy(object);
            auto connectedObjectNeedsEnergy = needsEnergy(connectedObject);
            auto connectedObjectLowEnergy = lowEnergy(object);

            auto flow = 0.0f;
            if (connectedObjectLowEnergy) {
                if (object->typeData.cell.usableEnergy > connectedObject->typeData.cell.usableEnergy) {
                    flow = (object->typeData.cell.usableEnergy - connectedObject->typeData.cell.usableEnergy) / 2;
                }
            } else {
                if ((cellNeedsEnergy && connectedObjectNeedsEnergy) || (!cellNeedsEnergy && !connectedObjectNeedsEnergy)) {
                    if (object->typeData.cell.usableEnergy > connectedObject->typeData.cell.usableEnergy) {
                        flow = (object->typeData.cell.usableEnergy - connectedObject->typeData.cell.usableEnergy) / 2;
                    }
                }
                if (!cellNeedsEnergy && connectedObjectNeedsEnergy) {
                    if (object->typeData.cell.usableEnergy > cellNormalEnergy) {
                        flow = object->typeData.cell.usableEnergy - cellNormalEnergy;
                    }
                }
            }

            if (flow > 0) {
                flow = min(2.0f, flow);
                auto orig = atomicAdd(&object->typeData.cell.usableEnergy, -flow);
                if (orig < cellMinEnergy) {
                    atomicAdd(&object->typeData.cell.usableEnergy, flow);
                } else {
                    atomicAdd(&connectedObject->typeData.cell.usableEnergy, flow);
                }
            }
        }

        // Flow of raw energy
        {
            if (object->typeData.cell.cellState == CellState_Ready
                && connectedObject->typeData.cell.cellState == CellState_Ready
                && connectedObject->typeData.cell.rawEnergy < SimulationParameters::maxRawEnergyThresholdForConduction) {

                auto flow = 0.0f;
                auto maxFlow = 0.0f;
                if (connectedObject->typeData.cell.cellType == CellType_Digestor) {
                    maxFlow += connectedObject->typeData.cell.cellTypeData.digestor.rawEnergyConductivity
                        * cudaSimulationParameters.maxRawEnergyConductivity.value[connectedObject->color] * TIMESTEPS_PER_CELL_FUNCTION;
                    if (object->typeData.cell.cellType == CellType_Digestor) {
                        maxFlow += object->typeData.cell.cellTypeData.digestor.rawEnergyConductivity
                            * cudaSimulationParameters.maxRawEnergyConductivity.value[object->color] * TIMESTEPS_PER_CELL_FUNCTION;

                        auto cellConversionRate = 1.0f - object->typeData.cell.cellTypeData.digestor.rawEnergyConductivity;
                        auto connectedObjectConversionRate = 1.0f - connectedObject->typeData.cell.cellTypeData.digestor.rawEnergyConductivity;
                        if (cellConversionRate == 0 && connectedObjectConversionRate == 0) {
                            cellConversionRate = 1;
                            connectedObjectConversionRate = 1;
                        }

                        auto targetEnergy = (object->typeData.cell.rawEnergy + connectedObject->typeData.cell.rawEnergy) * cellConversionRate
                            / (cellConversionRate + connectedObjectConversionRate);
                        flow = max(object->typeData.cell.rawEnergy - targetEnergy, 0.0f);
                    } else {
                        flow = object->typeData.cell.rawEnergy;
                    }
                }

                flow = min(maxFlow, flow);
                auto orig = atomicAdd(&object->typeData.cell.rawEnergy, -flow);
                if (orig < 0) {
                    atomicAdd(&object->typeData.cell.rawEnergy, flow);
                } else {
                    atomicAdd(&connectedObject->typeData.cell.rawEnergy, flow);
                }
            }
        }
    }
}

__device__ __inline__ float CellProcessor::getInitialAngelSpan(Object* object, int connectionIndex1, int connectionIndex2)
{
    if ((connectionIndex1 - connectionIndex2 + object->numConnections) % object->numConnections == 0) {
        return 0;
    }
    auto result = 0.0f;
    for (int i = connectionIndex1 + 1; i < connectionIndex1 + object->numConnections; i++) {
        auto index = i % object->numConnections;
        result += MuscleProcessor::getInitialAngleFromPrevious(object, index);
        if (index == connectionIndex2) {
            break;
        }
    }
    return Math::getNormalizedAngle(result, -180.0f);
}

__device__ __inline__ float CellProcessor::getInitialAngelSpan(Object* object, Object* connectedObject1, Object* connectedObject2)
{
    auto connectionIndex1 = -1;
    auto connectionIndex2 = -1;
    for (int i = 0; i < object->numConnections; i++) {
        if (object->connections[i].object == connectedObject1) {
            connectionIndex1 = i;
        }
        if (object->connections[i].object == connectedObject2) {
            connectionIndex2 = i;
        }
    }
    if (connectionIndex1 == -1 || connectionIndex2 == -1) {
        return 0;
    }
    return getInitialAngelSpan(object, connectionIndex1, connectionIndex2);
}

__inline__ __device__ void CellProcessor::decay(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }

        if (object->type == ObjectType_Cell) {
            if (object->typeData.cell.cellState == CellState_InstantDying) {
                ObjectConnectionProcessor::scheduleDeleteObject(data, index);
            } else {
                auto minCellEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);

                if (object->typeData.cell.cellState == CellState_Dying) {
                    auto cellDeathProbability =
                        ParameterCalculator::calcParameter(cudaSimulationParameters.cellDeathProbability, data, object->pos, object->color);
                    if (data.primaryNumberGen.random() <= cellDeathProbability) {
                        ObjectConnectionProcessor::scheduleDeleteObject(data, index);
                    }
                }

                bool cellDestruction = false;
                if (object->typeData.cell.usableEnergy < minCellEnergy) {
                    cellDestruction = true;
                }

                // Cell age radiation
                auto cellMaxAge = cudaSimulationParameters.maxCellAge.value[object->color];
                if (cellMaxAge > 0 && object->typeData.cell.age > cellMaxAge) {
                    cellDestruction = true;
                }

                if (cellDestruction) {
                    atomicExch(&object->typeData.cell.cellState, CellState_Dying);
                }
            }
        }
        if (object->type == ObjectType_FreeCell) {
            auto minCellEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);

            bool objectDestruction = false;
            if (object->typeData.freeCell.energy < minCellEnergy) {
                auto cellDeathProbability = ParameterCalculator::calcParameter(cudaSimulationParameters.cellDeathProbability, data, object->pos, object->color);
                if (data.primaryNumberGen.random() <= cellDeathProbability) {
                    objectDestruction = true;
                }
            }

            // Free cell age radiation
            auto cellMaxAge = Infinity<int>::value;
            if (object->type == ObjectType_FreeCell && cudaSimulationParameters.cellAgeLimiterToggle.value) {
                cellMaxAge = cudaSimulationParameters.freeCellMaxAge.value[object->color];
            }
            if (cellMaxAge > 0 && object->typeData.freeCell.age > cellMaxAge) {
                objectDestruction = true;
            }

            if (objectDestruction) {
                ObjectConnectionProcessor::scheduleDeleteObject(data, index);
            }
        }
    }
}
