#pragma once
#include "Math.cuh"
#include "ModelBasic/ElementaryTypes.h"
#include "QuantityConverter.cuh"
#include "SimulationData.cuh"

class ConstructorFunction
{
public:
    __inline__ __device__ static void processing(Token* token, SimulationData* data);

private:
    struct ClusterComponent
    {
        enum Type
        {
            Constructor = 0,
            ConstructionSite = 1
        };
    };
    struct Angles
    {
        float constructor;
        float constructionSite;
    };
    struct AngularMasses
    {
        float constructor;
        float constructionSite;
    };
    struct RotationMatrices
    {
        float constructor[2][2];
        float constructionSite[2][2];
    };

    __inline__ __device__ static bool checkDistance(float distance);
    __inline__ __device__ static Cell* getConstructionSite(Cell* cell);

    __inline__ __device__ static void continueConstruction(Token* token, Cell* constructionCell, SimulationData* data);
    __inline__ __device__ static void startNewConstruction();

    __inline__ __device__ static void continueConstructionWithRotationOnly(
        Token* token,
        Cell* constructionCell,
        Angles const& anglesToRotate,
        float desiredAngle,
        SimulationData* data);

    __inline__ __device__ static void continueConstructionWithRotationAndCreation(
        Token* token,
        Cell* constructionCell,
        Angles const& anglesToRotate,
        float desiredAngle,
        SimulationData* data);

    /*
    __inline__ __device__ static void removeConnection(Cell* cell, Cell* otherCell);
*/
    __inline__ __device__ static void
    tagConstructionSite(Cell* baseCell, Cell* constructionCell, Cell** cellArray1, Cell** cellArray2);

    __inline__ __device__ static Angles calcMinimalAngles(Cluster* cluster, Cell* constructionCell);
    __inline__ __device__ static AngularMasses calcAngularMasses(Cluster* cluster, Cell* constructionCell);
    __inline__ __device__ static RotationMatrices calcRotationMatrices(Angles const& angles);

    __inline__ __device__ static Angles calcAnglesToRotate(
        AngularMasses const& angularMasses,
        float desiredAngleBetweenConstructurAndConstructionSite);
    __inline__ __device__ static bool restrictAngles(Angles& angles, Angles const& minAngles);

    __inline__ __device__ static float
    calcAngularMassAfterRotation(Cluster* cluster, float2 const& centerOfRotation, Angles const& angles);
    __inline__ __device__ static void
    rotateClusterComponents(Cluster* cluster, float2 const& centerOfRotation, Angles const& angles);

    __inline__ __device__ static float2
    getRotateCellRelPos(Cell* cell, float2 const& centerOfRotation, RotationMatrices const& matrices);

    __inline__ __device__ static bool isObstaclePresent(
        bool ignoreOwnCluster,
        Cluster* cluster,
        float2 const& centerOfRotation,
        Angles const& anglesToRotate,
        Map<Cell> const& map);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ConstructorFunction::processing(Token* token, SimulationData* data)
{
    auto const command = token->memory[Enums::Constr::IN] % Enums::ConstrIn::_COUNTER;

    if (Enums::ConstrIn::DO_NOTHING == command) {
        token->memory[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS;
        return;
    }

    //TODO: short energy check for optimization

    auto const distance = QuantityConverter::convertDataToDistance(token->memory[Enums::Constr::IN_DIST]);
    if (!checkDistance(distance)) {
        token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_DIST;
    }

    auto const& cell = token->cell;
    auto const constructionSite = getConstructionSite(cell);

    if (constructionSite) {
        continueConstruction(token, constructionSite, data);
    } else {
        startNewConstruction();
    }
}

__inline__ __device__ bool ConstructorFunction::checkDistance(float distance)
{
    return distance > cudaSimulationParameters.cellMaxDistance;
}

__inline__ __device__ Cell* ConstructorFunction::getConstructionSite(Cell* cell)
{
    Cell* result = nullptr;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto const& connectingCell = cell->connections[i];
        if (connectingCell->tokenBlocked) {
            result = connectingCell;
        }
    }
    return result;
}

__inline__ __device__ void
ConstructorFunction::continueConstruction(Token* token, Cell* constructionCell, SimulationData* data)
{
    auto const& cell = token->cell;
    auto const& cluster = constructionCell->cluster;
    auto const cellArray1 = data->entities.cellPointers.getNewSubarray(cluster->numCellPointers + 1);
    auto const cellArray2 = data->entities.cellPointers.getNewSubarray(cluster->numCellPointers);
    tagConstructionSite(cell, constructionCell, cellArray1, cellArray2);

    if (ClusterComponent::ConstructionSite == cell->tag) {
        token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_CONNECTION;
        return;
    }

    auto const minAngles = calcMinimalAngles(cluster, constructionCell);
    auto const angularMasses = calcAngularMasses(cluster, constructionCell);
    auto const desiredAngleBetweenConstructurAndConstructionSite =
        QuantityConverter::convertDataToAngle(token->memory[Enums::Constr::INOUT_ANGLE]);

    auto anglesToRotate = calcAnglesToRotate(angularMasses, desiredAngleBetweenConstructurAndConstructionSite);
    auto const angleRestricted = restrictAngles(anglesToRotate, minAngles);

    if (angleRestricted) {
        continueConstructionWithRotationOnly(
            token, constructionCell, anglesToRotate, desiredAngleBetweenConstructurAndConstructionSite, data);
    } else {
        continueConstructionWithRotationAndCreation(
            token, constructionCell, anglesToRotate, desiredAngleBetweenConstructurAndConstructionSite, data);
    }
}

__inline__ __device__ void ConstructorFunction::startNewConstruction() {}

__inline__ __device__ void ConstructorFunction::continueConstructionWithRotationOnly(
    Token* token,
    Cell* constructionCell,
    Angles const& anglesToRotate,
    float desiredAngle,
    SimulationData* data)
{
    auto const& cluster = constructionCell->cluster;
    auto const kineticEnergyBeforeRotation =
        Physics::kineticEnergy(cluster->numCellPointers, cluster->vel, cluster->angularMass, cluster->angularVel);

    auto const angularMassAfterRotation =
        calcAngularMassAfterRotation(cluster, constructionCell->relPos, anglesToRotate);
    auto const angularVelAfterRotation =
        Physics::angularVelocity(cluster->angularMass, angularMassAfterRotation, cluster->angularVel);
    auto const kineticEnergyAfterRotation = Physics::kineticEnergy(
        cluster->numCellPointers, cluster->vel, angularMassAfterRotation, angularVelAfterRotation);

    auto const kineticEnergyDiff = kineticEnergyAfterRotation - kineticEnergyBeforeRotation;

    if (token->energy <= cudaSimulationParameters.cellFunctionConstructorOffspringCellEnergy
            + cudaSimulationParameters.tokenMinEnergy + kineticEnergyDiff) {
        token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_NO_ENERGY;
        return;
    }

    auto const command = token->memory[Enums::Constr::IN] % Enums::ConstrIn::_COUNTER;
    if (Enums::ConstrIn::SAFE == command || Enums::ConstrIn::UNSAFE == command) {
        auto ignoreOwnCluster = (Enums::ConstrIn::UNSAFE == command);
        if (isObstaclePresent(ignoreOwnCluster, cluster, constructionCell->relPos, anglesToRotate, data->cellMap)) {
            token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_OBSTACLE;
            return;
        }
    }

    rotateClusterComponents(cluster, constructionCell->relPos, anglesToRotate);

    token->memory[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS_ROT;
    token->memory[Enums::Constr::INOUT_ANGLE] = QuantityConverter::convertAngleToData(
        desiredAngle - (anglesToRotate.constructionSite - anglesToRotate.constructor));
}

__inline__ __device__ void ConstructorFunction::continueConstructionWithRotationAndCreation(
    Token* token,
    Cell* constructionCell,
    Angles const& anglesToRotate,
    float desiredAngle,
    SimulationData* data)
{
    auto const& cell = token->cell;
    auto const distance = QuantityConverter::convertDataToDistance(token->memory[Enums::Constr::IN_DIST]);

    auto displacementForConstructionSite = Math::normalized(constructionCell->relPos - cell->relPos) * distance;
    auto const option = token->memory[Enums::Constr::IN_OPTION] % Enums::ConstrInOption::_COUNTER;
    auto relPosOfNewCell = constructionCell->relPos;
    if (Enums::ConstrInOption::FINISH_WITH_SEP == option 
        || Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option) {
        relPosOfNewCell = relPosOfNewCell + displacementForConstructionSite;
        displacementForConstructionSite = displacementForConstructionSite * 2;
    }
}

/*
__inline__ __device__ void ConstructorFunction::removeConnection(Cell* cell, Cell* otherCell)
{
    auto remove = [](Cell* cell, Cell* connectionToRemove) {
        bool connectionFound = false;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto& connectingCell = cell->connections[i];
            if (!connectionFound) {
                if (connectingCell == connectionToRemove) {
                    connectionFound = true;
                }
            } else {
                cell->connections[i - 1] = connectingCell;
            }
        }
        --cell->numConnections;
    };

    remove(cell, otherCell);
    remove(otherCell, cell);
}*/

__inline__ __device__ void
ConstructorFunction::tagConstructionSite(Cell* baseCell, Cell* constructionCell, Cell** cells, Cell** otherCells)
{
    auto const& cluster = constructionCell->cluster;

    for (int cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
        auto const& otherCell = cluster->cellPointers[cellIndex];
        otherCell->tag = ClusterComponent::Constructor;
    }
    constructionCell->tag = ClusterComponent::ConstructionSite;

    cells[0] = constructionCell;
    int numElements = 1;
    int numOtherElements = 0;
    do {
        for (int cellIndex = 0; cellIndex < numElements; ++cellIndex) {
            auto& cell = cells[cellIndex];
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& connectingCell = cell->connections[i];
                if (cell == constructionCell
                    && connectingCell == baseCell) {  //ignore connection between baseCell and constructionCell
                    continue;
                }
                if (ClusterComponent::Constructor == connectingCell->tag) {
                    otherCells[numOtherElements++] = connectingCell;
                }
                connectingCell->tag = ClusterComponent::ConstructionSite;
            }
        }
        numElements = numOtherElements;
        numOtherElements = 0;
        cells = otherCells;
    } while (numElements > 0);
}

__inline__ __device__ auto ConstructorFunction::calcMinimalAngles(Cluster* cluster, Cell* constructionCell) -> Angles
{
    Angles result{360.0f, 360.0f};
    for (int cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        auto r = Math::length(cell->relPos - constructionCell->relPos);
        if (cudaSimulationParameters.cellMaxDistance < 2 * r) {
            auto a = abs(2.0 * asinf(cudaSimulationParameters.cellMaxDistance / (2.0 * r)) * RAD_TO_DEG);
            if (ClusterComponent::Constructor == cell->tag) {
                result.constructionSite = min(result.constructionSite, a);
            }
            if (ClusterComponent::ConstructionSite == cell->tag) {
                result.constructor = min(result.constructor, a);
            }
        }
    }
    return result;
}

__inline__ __device__ auto ConstructorFunction::calcAngularMasses(Cluster* cluster, Cell* constructionCell)
    -> AngularMasses
{
    AngularMasses result{0, 0};
    for (int cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        if (ClusterComponent::Constructor == cell->tag) {
            result.constructor =
                result.constructor + Math::lengthSquared(cell->relPos - constructionCell->relPos);
        }
        if (ClusterComponent::ConstructionSite == cell->tag) {
            result.constructionSite =
                result.constructionSite + Math::lengthSquared(cell->relPos - constructionCell->relPos);
        }
    }
    return result;
}

__inline__ __device__ auto ConstructorFunction::calcRotationMatrices(Angles const& angles) -> RotationMatrices
{
    RotationMatrices result;
    Math::rotationMatrix(angles.constructionSite, result.constructionSite);
    Math::rotationMatrix(angles.constructor, result.constructor);
    return result;
}

__inline__ __device__ auto ConstructorFunction::calcAnglesToRotate(
    AngularMasses const& angularMasses,
    float desiredAngleBetweenConstructurAndConstructionSite) -> Angles
{
    Angles result;
    auto sumAngularMasses = angularMasses.constructor + angularMasses.constructionSite;
    result.constructionSite =
        angularMasses.constructor * desiredAngleBetweenConstructurAndConstructionSite / sumAngularMasses;
    result.constructor =
        -angularMasses.constructionSite * desiredAngleBetweenConstructurAndConstructionSite / sumAngularMasses;

    return result;
}

__inline__ __device__ bool ConstructorFunction::restrictAngles(Angles& angles, Angles const& minAngles)
{
    auto result = false;
    if (abs(angles.constructionSite) > minAngles.constructionSite) {
        result = true;
        if (angles.constructionSite >= 0.0) {
            angles.constructionSite = abs(minAngles.constructionSite);
        }
        if (angles.constructionSite < 0.0) {
            angles.constructionSite = -abs(minAngles.constructionSite);
        }
    }
    if (abs(angles.constructor) > minAngles.constructor) {
        result = true;
        if (angles.constructor >= 0.0) {
            angles.constructor = abs(minAngles.constructor);
        }
        if (angles.constructor < 0.0) {
            angles.constructor = -abs(minAngles.constructor);
        }
    }
    return result;
}

__inline__ __device__ float ConstructorFunction::calcAngularMassAfterRotation(
    Cluster* cluster,
    float2 const& centerOfRotation,
    Angles const& angles)
{
    auto const matrices = calcRotationMatrices(angles);

    auto center = float2{0, 0};
    for (auto cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        auto const rotatedCellRelPos = getRotateCellRelPos(cell, centerOfRotation, matrices);
        center = center + rotatedCellRelPos;
    }
    center = center / cluster->numCellPointers;

    auto result = 0.0f;
    for (auto cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        auto const rotatedCellRelPos = getRotateCellRelPos(cell, centerOfRotation, matrices);
        auto const displacement = rotatedCellRelPos - center;
        result += Math::lengthSquared(displacement);
    }
    return result;
}

__inline__ __device__ void
ConstructorFunction::rotateClusterComponents(Cluster* cluster, float2 const& centerOfRotation, Angles const& angles)
{
    RotationMatrices matrices = calcRotationMatrices(angles);
    for (int cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        cell->relPos = getRotateCellRelPos(cell, centerOfRotation, matrices);
    }
}

__inline__ __device__ float2
ConstructorFunction::getRotateCellRelPos(Cell* cell, float2 const& centerOfRotation, RotationMatrices const& matrices)
{
    if (ClusterComponent::Constructor == cell->tag) {
        return Math::applyMatrix(cell->relPos - centerOfRotation, matrices.constructor) + centerOfRotation;
    }
    else {
        return Math::applyMatrix(cell->relPos - centerOfRotation, matrices.constructionSite) + centerOfRotation;
    }
}

__inline__ __device__ bool ConstructorFunction::isObstaclePresent(
    bool ignoreOwnCluster,
    Cluster* cluster,
    float2 const& centerOfRotation,
    Angles const& anglesToRotate,
    Map<Cell> const& map)
{
    RotationMatrices const matrices = calcRotationMatrices(anglesToRotate);
    Math::Matrix clusterMatrix;
    Math::rotationMatrix(cluster->angle, clusterMatrix);

    for (int cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        auto const relPos = getRotateCellRelPos(cell, centerOfRotation, matrices);

        auto const absPos = cluster->pos + Math::applyMatrix(relPos, clusterMatrix);
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                float2 const lookupPos = {absPos.x + dx, absPos.y + dy};
                if (auto const otherCell = map.get(lookupPos)) {
                    if (map.mapDistance(otherCell->absPos, absPos) >= cudaSimulationParameters.cellMinDistance) {
                        continue;
                    }
                    if (ignoreOwnCluster) {
                        if (cluster != otherCell->cluster) {
                            return true;
                        }
                    } else {
                        if (cell != otherCell) {
                            return true;
                        }
                    }

                    //check also connected cells
                    for (int i = 0; i < otherCell->numConnections; ++i) {
                        auto const connectedOtherCell = otherCell->connections[i];
                        if (map.mapDistance(connectedOtherCell->absPos, absPos)
                            >= cudaSimulationParameters.cellMinDistance) {
                            continue;
                        }
                        if (ignoreOwnCluster) {
                            if (cluster != connectedOtherCell->cluster) {
                                return true;
                            }
                        } else {
                            if (cell != connectedOtherCell) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}
