#pragma once
#include "HashMap.cuh"
#include "Math.cuh"
#include "ModelBasic/ElementaryTypes.h"
#include "QuantityConverter.cuh"
#include "SimulationData.cuh"

class ConstructorFunction
{
public:
    __inline__ __device__ void init_blockCall(Token* token, SimulationData* data);
    __inline__ __device__ void processing();

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
    struct EnergyForNewEntities
    {
        bool energyAvailable;
        float cell;
        float token;
    };

    __inline__ __device__ void checkMaxRadius(bool& result);
    __inline__ __device__ bool checkDistance(float distance);
    __inline__ __device__ Cell* getConstructionSite();

    __inline__ __device__ void continueConstruction(Cell* firstCellOfConstructionSite);
    __inline__ __device__ void startNewConstruction();

    __inline__ __device__ void
    continueConstructionWithRotationOnly(Cell* constructionCell, Angles const& anglesToRotate, float desiredAngle);

    __inline__ __device__ void continueConstructionWithRotationAndCreation(
        Cell* constructionCell,
        Angles const& anglesToRotate,
        float desiredAngle);

    __inline__ __device__ void tagConstructionSite(Cell* baseCell, Cell* firstCellOfConstructionSite);

    __inline__ __device__ void calcMaxAngles(Cell* constructionCell, Angles& result);
    __inline__ __device__ void calcAngularMasses(Cluster* cluster, Cell* constructionCell, AngularMasses& result);
    __inline__ __device__ RotationMatrices calcRotationMatrices(Angles const& angles);

    __inline__ __device__ float calcFreeAngle(Cell* cell);
    __inline__ __device__ Angles
    calcAnglesToRotate(AngularMasses const& angularMasses, float desiredAngleBetweenConstructurAndConstructionSite);
    __inline__ __device__ bool restrictAngles(Angles& angles, Angles const& minAngles);

    __inline__ __device__ void calcAngularMassAfterTransformationAndAddingCell(
        float2 const& relPosOfNewCell,
        float2 const& centerOfRotation,
        RotationMatrices const& rotationMatrices,
        float2 const& displacementOfConstructionSite,
        float& result);
    __inline__ __device__ void calcAngularMassAfterAddingCell(Cluster* cluster, float2 const& relPosOfNewCell, float& result);
    __inline__ __device__ EnergyForNewEntities adaptEnergies(Token* token, float energyLoss);


    __inline__ __device__ void transformClusterComponents(
        float2 const& centerOfRotation,
        RotationMatrices const& rotationMatrices,
        float2 const& displacementForConstructionSite);
    __inline__ __device__ void adaptRelPositions();
    __inline__ __device__ void completeCellAbsPosAndVel();

    __inline__ __device__ float2 getTransformedCellRelPos(
        Cell* cell,
        float2 const& centerOfRotation,
        RotationMatrices const& matrices,
        float2 const& displacementForConstructionSite);

    struct CellAndNewAbsPos
    {
        Cell* cell;
        float2 newAbsPos;
    };
    __inline__ __device__ void isObstaclePresent_onlyRotation(
        bool ignoreOwnCluster,
        float2 const& centerOfRotation,
        RotationMatrices const& rotationMatrices,
        bool& result);
    __inline__ __device__ void isObstaclePresent_rotationAndCreation(
        bool ignoreOwnCluster,
        float2 const& relPosOfNewCell,
        float2 const& centerOfRotation,
        RotationMatrices const& rotationMatrices,
        float2 const& displacementOfConstructionSite,
        bool& result);
    __inline__ __device__ void isObstaclePresent_firstCreation(
        bool ignoreOwnCluster,
        Cluster* cluster,
        float2 const& relPosOfNewCell,
        Map<Cell> const& map,
        bool& result);
    __inline__ __device__ bool isObstaclePresent_helper(
        bool ignoreOwnCluster,
        Cluster* cluster,
        Cell* cell,
        float2 const& absPos,
        Map<Cell> const& map,
        HashMap<int2, CellAndNewAbsPos>& tempMap);

    __inline__ __device__ void
    constructNewCell(float2 const& relPosOfNewCell, float const energyOfNewCell, Cell*& result);
    __inline__ __device__ void constructNewToken(
        Cell* cellOfNewToken,
        Cell* sourceCellOfNewToken,
        float energyOfNewToken,
        bool duplicate,
        Token*& result);

    __inline__ __device__ void addCellToCluster(Cell* newCell, Cell** newCellPointers);
    __inline__ __device__ void addTokenToCluster(Token* token, Token** newTokenPointers);

    __inline__ __device__ void separateConstructionWhenFinished(Cell* newCell);

    __inline__ __device__ void connectNewCell(Cell* newCell, Cell* cellOfConstructionSite);
    __inline__ __device__ void removeConnection(Cell* cell1, Cell* cell2);
    enum class AdaptMaxConnections
    {
        No,
        Yes
    };
    __inline__ __device__ AdaptMaxConnections isAdaptMaxConnections(Token* token);
    __inline__ __device__ bool isConnectable(int numConnections, int maxConnections, AdaptMaxConnections adaptMaxConnections);
    __inline__ __device__ void establishConnection(Cell* cell1, Cell* cell2, AdaptMaxConnections adaptMaxConnections);

private:
    SimulationData* _data;
    Token* _token;
    Cluster* _cluster;
    PartitionData _cellBlock;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ConstructorFunction::processing()
{
    auto const command = _token->memory[Enums::Constr::IN] % Enums::ConstrIn::_COUNTER;
    if (Enums::ConstrIn::DO_NOTHING == command) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS;
        __syncthreads();
        return;
    }

    __shared__ bool isRadiusTooLarge;
    checkMaxRadius(isRadiusTooLarge);
    __syncthreads();

    if (!isRadiusTooLarge) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_MAX_RADIUS;
        __syncthreads();
        return;
    }

    //TODO: short energy check for optimization

    __shared__ Cell* firstCellOfConstructionSite;
    if (0 == threadIdx.x) {
        firstCellOfConstructionSite = getConstructionSite();
    }
    __syncthreads();

    if (firstCellOfConstructionSite) {
        auto const distance = QuantityConverter::convertDataToDistance(_token->memory[Enums::Constr::IN_DIST]);
        if (!checkDistance(distance)) {
            _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_DIST;
            __syncthreads();
            return;
        }
        __syncthreads();
        continueConstruction(firstCellOfConstructionSite);
    } else {
        startNewConstruction();
    }
    __syncthreads();
}

__inline__ __device__ void ConstructorFunction::init_blockCall(Token* token, SimulationData* data)
{
    _data = data;
    _cluster = token->cell->cluster;
    _token = token;
    _cellBlock = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);
}

__inline__ __device__ void ConstructorFunction::checkMaxRadius(bool& result)
{
    __shared__ float maxRadius;
    if (0 == threadIdx.x) {
        result = true;
        maxRadius = _data->cellMap.getMaxRadius();
    }
    __syncthreads();
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        if (Math::length(cell->relPos) >= maxRadius - FP_PRECISION) {
            result = false;
            return;
        }
    }
}

__inline__ __device__ bool ConstructorFunction::checkDistance(float distance)
{
    return cudaSimulationParameters.cellMinDistance < distance && distance < cudaSimulationParameters.cellMaxDistance;
}

__inline__ __device__ Cell* ConstructorFunction::getConstructionSite()
{
    Cell* result = nullptr;
    auto const& cell = _token->cell;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto const& connectingCell = cell->connections[i];
        if (connectingCell->tokenBlocked) {
            result = connectingCell;
        }
    }
    return result;
}

__inline__ __device__ void ConstructorFunction::continueConstruction(Cell* firstCellOfConstructionSite)
{
    auto const& cell = _token->cell;
    tagConstructionSite(cell, firstCellOfConstructionSite);
    __syncthreads();

    if (ClusterComponent::ConstructionSite == cell->tag) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_CONNECTION;
        __syncthreads();
        return;
    }

    __shared__ Angles maxAngles;
    calcMaxAngles(firstCellOfConstructionSite, maxAngles);
    __syncthreads();

    __shared__ AngularMasses angularMasses;
    calcAngularMasses(_cluster, firstCellOfConstructionSite, angularMasses);
    __syncthreads();

    __shared__ float desiredAngleBetweenConstructurAndConstructionSite;
    __shared__ Angles anglesToRotate;
    __shared__ bool isAngleRestricted;
    if (0 == threadIdx.x) {
        desiredAngleBetweenConstructurAndConstructionSite =
            QuantityConverter::convertDataToAngle(_token->memory[Enums::Constr::INOUT_ANGLE]);
        anglesToRotate = calcAnglesToRotate(angularMasses, desiredAngleBetweenConstructurAndConstructionSite);
        isAngleRestricted = restrictAngles(anglesToRotate, maxAngles);
    }
    __syncthreads();

    if (isAngleRestricted) {

        //angle discretization correction
        if (0 == threadIdx.x) {
            anglesToRotate.constructor =
                QuantityConverter::convertDataToAngle(QuantityConverter::convertAngleToData(anglesToRotate.constructor));
            anglesToRotate.constructionSite = QuantityConverter::convertDataToAngle(
                QuantityConverter::convertAngleToData(anglesToRotate.constructionSite));
        }
        __syncthreads();

        continueConstructionWithRotationOnly(
            firstCellOfConstructionSite,
            anglesToRotate,
            desiredAngleBetweenConstructurAndConstructionSite);
    }
    else {
        continueConstructionWithRotationAndCreation(
            firstCellOfConstructionSite,
            anglesToRotate,
            desiredAngleBetweenConstructurAndConstructionSite);
    }
    __syncthreads();
}

__inline__ __device__ void ConstructorFunction::startNewConstruction()
{
    auto const& cell = _token->cell;
    auto const adaptMaxConnections = isAdaptMaxConnections(_token);

    if (!isConnectable(cell->numConnections, cell->maxConnections, adaptMaxConnections)) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_CONNECTION;
        return;
    }

    __shared__ float2 relPosOfNewCell;
    __shared__ Enums::ConstrIn::Type command;
    __shared__ Enums::ConstrInOption::Type option;
    if (0 == threadIdx.x) {
        auto const freeAngle = calcFreeAngle(cell);
        auto const newCellAngle =
            QuantityConverter::convertDataToAngle(_token->memory[Enums::Constr::INOUT_ANGLE]) + freeAngle;

        command = static_cast<Enums::ConstrIn::Type>(_token->memory[Enums::Constr::IN] % Enums::ConstrIn::_COUNTER);
        option = static_cast<Enums::ConstrInOption::Type>(
            _token->memory[Enums::Constr::IN_OPTION] % Enums::ConstrInOption::_COUNTER);
        bool const separation = Enums::ConstrInOption::FINISH_WITH_SEP == option
            || Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
            || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option;

        auto const distance = QuantityConverter::convertDataToDistance(_token->memory[Enums::Constr::IN_DIST]);
        auto const relPosOfNewCellDelta =
            Math::unitVectorOfAngle(newCellAngle) * cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance;
        relPosOfNewCell = separation
            ? cell->relPos + relPosOfNewCellDelta + Math::unitVectorOfAngle(newCellAngle) * distance
            : cell->relPos + relPosOfNewCellDelta;

    }
    __syncthreads();

    if (Enums::ConstrIn::SAFE == command || Enums::ConstrIn::UNSAFE == command) {
        auto ignoreOwnCluster = (Enums::ConstrIn::UNSAFE == command);
        __shared__ bool isObstaclePresent;
        isObstaclePresent_firstCreation(ignoreOwnCluster, _cluster, relPosOfNewCell, _data->cellMap, isObstaclePresent);
        __syncthreads();

        if (isObstaclePresent) {
            _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_OBSTACLE;
            return;
        }
    }

    __shared__ float angularMassAfterRotation;
    __shared__ float angularVelAfterRotation;
    __shared__ float kineticEnergyBeforeRotation;
    __shared__ EnergyForNewEntities energyForNewEntities;
    if (0 == threadIdx.x) {
        kineticEnergyBeforeRotation =
            Physics::kineticEnergy(_cluster->numCellPointers, _cluster->vel, _cluster->angularMass, _cluster->angularVel);
    }
    __syncthreads();

    calcAngularMassAfterAddingCell(_cluster, relPosOfNewCell, angularMassAfterRotation);
    __syncthreads();

    if (0 == threadIdx.x) {
        angularVelAfterRotation =
            Physics::angularVelocity(_cluster->angularMass, angularMassAfterRotation, _cluster->angularVel);
        auto const kineticEnergyAfterRotation = Physics::kineticEnergy(
            _cluster->numCellPointers, _cluster->vel, angularMassAfterRotation, angularVelAfterRotation);
        auto const kineticEnergyDiff = kineticEnergyAfterRotation - kineticEnergyBeforeRotation;

        energyForNewEntities = adaptEnergies(_token, kineticEnergyDiff);
    }
    __syncthreads();

    if (!energyForNewEntities.energyAvailable) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_NO_ENERGY;
        return;
    }

    __shared__ Cell* newCell;
    constructNewCell(relPosOfNewCell, energyForNewEntities.cell, newCell);
    __syncthreads();

    __shared__ Cell** newCellPointers;
    if (0 == threadIdx.x) {
        newCellPointers = _data->entities.cellPointers.getNewSubarray(_cluster->numCellPointers + 1);
    }
    __syncthreads();

    addCellToCluster(newCell, newCellPointers);
    __syncthreads();

    _cellBlock = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);

    if (0 == threadIdx.x) {
        establishConnection(newCell, cell, adaptMaxConnections);
    }
    __syncthreads();

    adaptRelPositions();
    __syncthreads();

    completeCellAbsPosAndVel();
    __syncthreads();

    __shared__ bool createEmptyToken;
    __shared__ bool createDuplicateToken;
    if (0 == threadIdx.x) {
        _cluster->angularVel = angularVelAfterRotation;
        _cluster->angularMass = angularMassAfterRotation;
        separateConstructionWhenFinished(newCell);
        createEmptyToken = Enums::ConstrInOption::CREATE_EMPTY_TOKEN == option
            || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option;
        createDuplicateToken = Enums::ConstrInOption::CREATE_DUP_TOKEN == option;
    }
    __syncthreads();

    if (createEmptyToken || createDuplicateToken) {
        
        __shared__ Token* newToken;
        constructNewToken(newCell, cell, energyForNewEntities.token, createDuplicateToken, newToken);
        __syncthreads();
            
        __shared__ Token** newTokenPointers;
        if (0 == threadIdx.x) {
            newTokenPointers = _data->entities.tokenPointers.getNewSubarray(_cluster->numTokenPointers + 1);
        }
        __syncthreads();

        addTokenToCluster(newToken, newTokenPointers);
        __syncthreads();
    }

    if (0 == threadIdx.x) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS;
        _token->memory[Enums::Constr::INOUT_ANGLE] = 0;
    }
}

__inline__ __device__ void ConstructorFunction::continueConstructionWithRotationOnly(
    Cell* firstCellOfConstructionSite,
    Angles const& anglesToRotate,
    float desiredAngle)
{
    __shared__ RotationMatrices rotationMatrices;
    __shared__ float kineticEnergyBeforeRotation;
    if (0 == threadIdx.x) {
        kineticEnergyBeforeRotation =
            Physics::kineticEnergy(_cluster->numCellPointers, _cluster->vel, _cluster->angularMass, _cluster->angularVel);
        rotationMatrices = calcRotationMatrices(anglesToRotate);
    }
    __syncthreads();

    __shared__ float angularMassAfterRotation;
    calcAngularMassAfterTransformationAndAddingCell(
        {0, 0}, firstCellOfConstructionSite->relPos, rotationMatrices, {0, 0}, angularMassAfterRotation);
    __syncthreads();

    __shared__ float angularVelAfterRotation;
    __shared__ float kineticEnergyDiff;
    if (0 == threadIdx.x) {
        angularVelAfterRotation =
            Physics::angularVelocity(_cluster->angularMass, angularMassAfterRotation, _cluster->angularVel);
        auto const kineticEnergyAfterRotation = Physics::kineticEnergy(
            _cluster->numCellPointers, _cluster->vel, angularMassAfterRotation, angularVelAfterRotation);

        kineticEnergyDiff = kineticEnergyAfterRotation - kineticEnergyBeforeRotation;
    }
    __syncthreads();

    if (_token->energy <= cudaSimulationParameters.cellFunctionConstructorOffspringCellEnergy
            + cudaSimulationParameters.tokenMinEnergy + kineticEnergyDiff) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_NO_ENERGY;
        __syncthreads();
        return;
    }

    auto const command = _token->memory[Enums::Constr::IN] % Enums::ConstrIn::_COUNTER;
    if (Enums::ConstrIn::SAFE == command || Enums::ConstrIn::UNSAFE == command) {
        auto const ignoreOwnCluster = (Enums::ConstrIn::UNSAFE == command);
        
        __shared__ bool result;
        isObstaclePresent_onlyRotation(
            ignoreOwnCluster, firstCellOfConstructionSite->relPos, rotationMatrices, result);
        __syncthreads();

        if (result) {
            _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_OBSTACLE;
            __syncthreads();
            return;
        }
    }

    transformClusterComponents(firstCellOfConstructionSite->relPos, rotationMatrices, { 0, 0 });
    __syncthreads();

    adaptRelPositions();
    __syncthreads();

    completeCellAbsPosAndVel();
    __syncthreads();

    if (0 == threadIdx.x) {
        _cluster->angularVel = angularVelAfterRotation;
        _cluster->angularMass = angularMassAfterRotation;

        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS_ROT;
        _token->memory[Enums::Constr::INOUT_ANGLE] = QuantityConverter::convertAngleToData(
            desiredAngle - (anglesToRotate.constructionSite - anglesToRotate.constructor));
    }
}

__inline__ __device__ void ConstructorFunction::continueConstructionWithRotationAndCreation(
    Cell* firstCellOfConstructionSite,
    Angles const& anglesToRotate,
    float desiredAngle)
{
    auto const& cell = _token->cell;

    auto const adaptMaxConnections = isAdaptMaxConnections(_token);
    if (1 == _token->memory[Enums::Constr::IN_CELL_MAX_CONNECTIONS]) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_CONNECTION;
        __syncthreads();
        return;
    }

    __shared__ float2 relPosOfNewCell;
    __shared__ float2 centerOfRotation;
    __shared__ RotationMatrices rotationMatrices;
    __shared__ float2 displacementForConstructionSite;
    __shared__ Enums::ConstrInOption::Type option;
    __shared__ Enums::ConstrIn::Type command;
    if (0 == threadIdx.x) {
        auto const distance = QuantityConverter::convertDataToDistance(_token->memory[Enums::Constr::IN_DIST]);

        relPosOfNewCell = firstCellOfConstructionSite->relPos;
        centerOfRotation = firstCellOfConstructionSite->relPos;
        rotationMatrices = calcRotationMatrices(anglesToRotate);

        auto const cellRelPos_transformed = getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, { 0, 0 });
 
        displacementForConstructionSite =
            Math::normalized(firstCellOfConstructionSite->relPos - cellRelPos_transformed) * distance;
        option = static_cast<Enums::ConstrInOption::Type>(
            _token->memory[Enums::Constr::IN_OPTION] % Enums::ConstrInOption::_COUNTER);
        if (Enums::ConstrInOption::FINISH_WITH_SEP == option || Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
            || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option) {

            relPosOfNewCell = relPosOfNewCell + displacementForConstructionSite;
            displacementForConstructionSite = displacementForConstructionSite * 2;
        }
        command = static_cast<Enums::ConstrIn::Type>(_token->memory[Enums::Constr::IN] % Enums::ConstrIn::_COUNTER);
    }
    __syncthreads();

    if (Enums::ConstrIn::SAFE == command || Enums::ConstrIn::UNSAFE == command) {
        
        auto const ignoreOwnCluster = (Enums::ConstrIn::UNSAFE == command);
        
        __shared__ bool result;
        isObstaclePresent_rotationAndCreation(
            ignoreOwnCluster,
            relPosOfNewCell,
            centerOfRotation,
            rotationMatrices,
            displacementForConstructionSite,
            result);
        __syncthreads();

        if (result) {
            _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_OBSTACLE;
            __syncthreads();
            return;
        }
    }

    __shared__ float kineticEnergyBeforeRotation;
    if (0 == threadIdx.x) {
        kineticEnergyBeforeRotation = Physics::kineticEnergy(
            _cluster->numCellPointers, _cluster->vel, _cluster->angularMass, _cluster->angularVel);
    }
    __syncthreads();

    __shared__ float angularMassAfterRotation;
    calcAngularMassAfterTransformationAndAddingCell(
        relPosOfNewCell,
        centerOfRotation,
        rotationMatrices,
        displacementForConstructionSite,
        angularMassAfterRotation);

    __shared__ float angularVelAfterRotation;
    __shared__ EnergyForNewEntities energyForNewEntities;
    if (0 == threadIdx.x) {
        angularVelAfterRotation =
            Physics::angularVelocity(_cluster->angularMass, angularMassAfterRotation, _cluster->angularVel);
        auto const kineticEnergyAfterRotation = Physics::kineticEnergy(
            _cluster->numCellPointers, _cluster->vel, angularMassAfterRotation, angularVelAfterRotation);
        auto const kineticEnergyDiff = kineticEnergyAfterRotation - kineticEnergyBeforeRotation;
        energyForNewEntities = adaptEnergies(_token, kineticEnergyDiff);
    }
    __syncthreads();

    if (!energyForNewEntities.energyAvailable) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_NO_ENERGY;
        __syncthreads();
        return;
    }

    transformClusterComponents(centerOfRotation, rotationMatrices, displacementForConstructionSite);

    __shared__ Cell* newCell;
    constructNewCell(relPosOfNewCell, energyForNewEntities.cell, newCell);
    __syncthreads();

    __shared__ Cell** newCellPointers;
    if (0 == threadIdx.x) {
        newCellPointers = _data->entities.cellPointers.getNewSubarray(_cluster->numCellPointers + 1);
    }
    __syncthreads();

    addCellToCluster(newCell, newCellPointers);
    __syncthreads();

    _cellBlock = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);

    connectNewCell(newCell, firstCellOfConstructionSite);
    __syncthreads();

    adaptRelPositions();
    __syncthreads();

    completeCellAbsPosAndVel();
    __syncthreads();

    __shared__ bool createEmptyToken;
    __shared__ bool createDuplicateToken;
    if (0 == threadIdx.x) {
        _cluster->angularVel = angularVelAfterRotation;
        _cluster->angularMass = angularMassAfterRotation;

        firstCellOfConstructionSite->tokenBlocked = false;  //disable token blocking on construction side
        separateConstructionWhenFinished(newCell);

        createEmptyToken = Enums::ConstrInOption::CREATE_EMPTY_TOKEN == option
            || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option;
        createDuplicateToken = Enums::ConstrInOption::CREATE_DUP_TOKEN == option;
    }
    __syncthreads();

    if (createEmptyToken || createDuplicateToken) {
        __shared__ Token* newToken;
        constructNewToken(newCell, cell, energyForNewEntities.token, createDuplicateToken, newToken);
        __syncthreads();

        __shared__ Token** newTokenPointers;
        if (0 == threadIdx.x) {
            newTokenPointers = _data->entities.tokenPointers.getNewSubarray(_cluster->numTokenPointers + 1);
        }
        __syncthreads();

        addTokenToCluster(newToken, newTokenPointers);
    }

    if (0 == threadIdx.x) {
        _token->memory[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS;
        _token->memory[Enums::Constr::INOUT_ANGLE] = 0;
    }
}

__inline__ __device__ auto ConstructorFunction::adaptEnergies(Token* token, float energyLoss) -> EnergyForNewEntities
{
    EnergyForNewEntities result;
    result.energyAvailable = true;
    result.token = 0.0f;
    result.cell = cudaSimulationParameters.cellFunctionConstructorOffspringCellEnergy;

    auto const& cell = token->cell;
    auto const option = token->memory[Enums::Constr::IN_OPTION] % Enums::ConstrInOption::_COUNTER;

    if (Enums::ConstrInOption::CREATE_EMPTY_TOKEN == option || Enums::ConstrInOption::CREATE_DUP_TOKEN == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option) {
        result.token = cudaSimulationParameters.cellFunctionConstructorOffspringTokenEnergy;
    }

    if (token->energy <= cudaSimulationParameters.cellFunctionConstructorOffspringCellEnergy + result.token + energyLoss
            + cudaSimulationParameters.tokenMinEnergy) {
        result.energyAvailable = false;
        return result;
    }

    token->energy -= cudaSimulationParameters.cellFunctionConstructorOffspringCellEnergy + result.token + energyLoss;
    if (Enums::ConstrInOption::CREATE_EMPTY_TOKEN == option || Enums::ConstrInOption::CREATE_DUP_TOKEN == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option) {
        auto const averageEnergy = (cell->energy + result.cell) / 2;
        cell->energy = averageEnergy;
        result.cell = averageEnergy;
    }

    return result;
}

__inline__ __device__ void ConstructorFunction::tagConstructionSite(Cell* baseCell, Cell* firstCellOfConstructionSite)
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto& cell = _cluster->cellPointers[cellIndex];
        cell->tag = ClusterComponent::Constructor;
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        firstCellOfConstructionSite->tag = ClusterComponent::ConstructionSite;
    }

    __shared__ bool changes;
    do {
        changes = false;
        __syncthreads();

        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto& cell = _cluster->cellPointers[cellIndex];
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& otherCell = cell->connections[i];
                if (otherCell->tag > cell->tag) {
                    if (cell == firstCellOfConstructionSite && otherCell == baseCell
                        || cell == baseCell && otherCell == firstCellOfConstructionSite) {
                        continue;
                    }
                    cell->tag = otherCell->tag;
                    changes = true;
                }
            }
        }
        __syncthreads();
    } while (changes);
}

__inline__ __device__ void ConstructorFunction::calcMaxAngles(Cell* constructionCell, Angles& result)
{
    if (0 == threadIdx.x) {
        result = { 360.0f, 360.0f };
    }
    __syncthreads();
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        auto r = Math::length(cell->relPos - constructionCell->relPos);
        if (cudaSimulationParameters.cellMaxDistance < 2 * r) {
            auto a = abs(2.0 * asinf(cudaSimulationParameters.cellMaxDistance / (2.0 * r)) * RAD_TO_DEG);
            if (ClusterComponent::Constructor == cell->tag) {
                result.constructor = min(result.constructor, a);
            }
            if (ClusterComponent::ConstructionSite == cell->tag) {
                result.constructionSite = min(result.constructionSite, a);
            }
        }
    }
}

__inline__ __device__ void ConstructorFunction::calcAngularMasses(Cluster* cluster, Cell* firstCellOfConstructionSite, AngularMasses& result)
{
    if (0 == threadIdx.x) {
        result = { 0, 0 };
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        auto const angularMassElement = Math::lengthSquared(cell->relPos - firstCellOfConstructionSite->relPos);
        if (ClusterComponent::Constructor == cell->tag) {
            atomicAdd_block(&result.constructor, angularMassElement);
        }
        if (ClusterComponent::ConstructionSite == cell->tag) {
            atomicAdd_block(&result.constructionSite, angularMassElement);
        }
    }
}

__inline__ __device__ auto ConstructorFunction::calcRotationMatrices(Angles const& angles) -> RotationMatrices
{
    RotationMatrices result;
    Math::rotationMatrix(angles.constructionSite, result.constructionSite);
    Math::rotationMatrix(angles.constructor, result.constructor);
    return result;
}

__inline__ __device__ float ConstructorFunction::calcFreeAngle(Cell* cell)
{
    auto const numConnections = cell->numConnections;
    float angles[MAX_CELL_BONDS];
    for (int i = 0; i < numConnections; ++i) {
        auto const displacement = cell->connections[i]->relPos - cell->relPos;
        auto const angleToAdd = Math::angleOfVector(displacement);
        auto indexToAdd = 0;
        for (; indexToAdd < i; ++indexToAdd) {
            if (angles[indexToAdd] > angleToAdd) {
                break;
            }
        }
        for (int j = indexToAdd; j < numConnections - 1; ++j) {
            angles[j + 1] = angles[j];
        }
        angles[indexToAdd] = angleToAdd;
    }

    auto largestAnglesDiff = 0.0f;
    auto result = 0.0f;
    for (int i = 0; i < numConnections; ++i) {
        auto angleDiff = angles[(i + 1) % numConnections] - angles[i];
        if (angleDiff <= 0.0f) {
            angleDiff += 360.0f;
        }
        if (angleDiff > 360.0f) {
            angleDiff -= 360.0f;
        }
        if (angleDiff > largestAnglesDiff) {
            largestAnglesDiff = angleDiff;
            result = angles[i] + angleDiff / 2;
        }
    }

    return result;
}

__inline__ __device__ auto ConstructorFunction::calcAnglesToRotate(
    AngularMasses const& angularMasses,
    float desiredAngleBetweenConstructurAndConstructionSite) -> Angles
{
    Angles result;
    auto const sumAngularMasses = angularMasses.constructor + angularMasses.constructionSite;
    result.constructionSite =
        angularMasses.constructor * desiredAngleBetweenConstructurAndConstructionSite / sumAngularMasses;
    result.constructor =
        -angularMasses.constructionSite * desiredAngleBetweenConstructurAndConstructionSite / sumAngularMasses;

    return result;
}

__inline__ __device__ bool ConstructorFunction::restrictAngles(Angles& angles, Angles const& maxAngles)
{
    auto result = false;
    if (abs(angles.constructionSite) > maxAngles.constructionSite) {
        result = true;
        if (angles.constructionSite >= 0.0) {
            angles.constructionSite = abs(maxAngles.constructionSite);
        }
        if (angles.constructionSite < 0.0) {
            angles.constructionSite = -abs(maxAngles.constructionSite);
        }
    }
    if (abs(angles.constructor) > maxAngles.constructor) {
        result = true;
        if (angles.constructor >= 0.0) {
            angles.constructor = abs(maxAngles.constructor);
        }
        if (angles.constructor < 0.0) {
            angles.constructor = -abs(maxAngles.constructor);
        }
    }
    return result;
}

__inline__ __device__ void ConstructorFunction::calcAngularMassAfterTransformationAndAddingCell(
    float2 const& relPosOfNewCell,
    float2 const& centerOfRotation,
    RotationMatrices const& rotationMatrices,
    float2 const& displacementOfConstructionSite,
    float& result)
{
    __shared__ float2 center;
    if (0 == threadIdx.x) {
        center = relPosOfNewCell;
    }
    __syncthreads();

    for (auto cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        auto cellRelPosTransformed =
            getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, displacementOfConstructionSite);
        atomicAdd_block(&center.x, cellRelPosTransformed.x);
        atomicAdd_block(&center.y, cellRelPosTransformed.y);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        center = center / _cluster->numCellPointers;
        result = Math::lengthSquared(relPosOfNewCell - center);
    }
    __syncthreads();

    for (auto cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        auto cellRelPosTransformed =
            getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, displacementOfConstructionSite);
        atomicAdd_block(&result, Math::lengthSquared(cellRelPosTransformed - center));
    }
    __syncthreads();
}

__inline__ __device__ void
ConstructorFunction::calcAngularMassAfterAddingCell(Cluster* cluster, float2 const& relPosOfNewCell, float& result)
{
    __shared__ float2 center;
    if (0 == threadIdx.x) {
        center = relPosOfNewCell;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        atomicAdd_block(&center.x, cell->relPos.x);
        atomicAdd_block(&center.y, cell->relPos.y);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        center = center / cluster->numCellPointers;
        result = Math::lengthSquared(relPosOfNewCell - center);
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        result += Math::lengthSquared(cell->relPos - center);
    }
}

__inline__ __device__ void ConstructorFunction::transformClusterComponents(
    float2 const& centerOfRotation,
    RotationMatrices const& rotationMatrices,
    float2 const& displacementForConstructionSite)
{
    float rotMatrix[2][2];
    Math::rotationMatrix(_cluster->angle, rotMatrix);
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex;  ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        cell->relPos =
            getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, displacementForConstructionSite);
        cell->absPos = Math::applyMatrix(cell->relPos, rotMatrix) + _cluster->pos;
    }
}

__inline__ __device__ void ConstructorFunction::adaptRelPositions()
{
    __shared__ float2 newCenter;
    if (0 == threadIdx.x) {
        newCenter = {0, 0};
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        atomicAdd_block(&newCenter.x, cell->relPos.x);
        atomicAdd_block(&newCenter.y, cell->relPos.y);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        newCenter = newCenter / _cluster->numCellPointers;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        cell->relPos = cell->relPos - newCenter;
    }
}

__inline__ __device__ void ConstructorFunction::completeCellAbsPosAndVel()
{
    Math::Matrix rotationMatrix;
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto& cell = _cluster->cellPointers[cellIndex];
        Math::rotationMatrix(_cluster->angle, rotationMatrix);
        cell->absPos = Math::applyMatrix(cell->relPos, rotationMatrix) + _cluster->pos;

        auto r = cell->absPos - _cluster->pos;
        cell->vel = Physics::tangentialVelocity(r, _cluster->vel, _cluster->angularVel);
    }
}

__inline__ __device__ float2 ConstructorFunction::getTransformedCellRelPos(
    Cell* cell,
    float2 const& centerOfRotation,
    RotationMatrices const& matrices,
    float2 const& displacementForConstructionSite)
{
    if (ClusterComponent::Constructor == cell->tag) {
        return Math::applyMatrix(cell->relPos - centerOfRotation, matrices.constructor) + centerOfRotation;
    }
    if (ClusterComponent::ConstructionSite == cell->tag) {
        return Math::applyMatrix(cell->relPos - centerOfRotation, matrices.constructionSite) + centerOfRotation
            + displacementForConstructionSite;
    }
    return cell->relPos;
}

__inline__ __device__ void ConstructorFunction::isObstaclePresent_onlyRotation(
    bool ignoreOwnCluster,
    float2 const& centerOfRotation,
    RotationMatrices const& rotationMatrices,
    bool& result)
{

    __shared__ HashMap<int2, CellAndNewAbsPos> tempCellMap;
    tempCellMap.init_blockCall(_cluster->numCellPointers * 2, _data->arrays);
    __syncthreads();

    __shared__ float2 newCenter;
    if (0 == threadIdx.x) {
        newCenter = { 0, 0 };
        result = false;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        auto relPos = getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, {0, 0});
        atomicAdd_block(&newCenter.x, relPos.x);
        atomicAdd_block(&newCenter.y, relPos.y);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        newCenter = newCenter / _cluster->numCellPointers;
    }
    __syncthreads();

    Math::Matrix clusterMatrix;
    Math::rotationMatrix(_cluster->angle, clusterMatrix);
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        auto relPos = getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, {0, 0});
        relPos = relPos - newCenter;
        auto const absPos = _cluster->pos + Math::applyMatrix(relPos, clusterMatrix);
        if (isObstaclePresent_helper(ignoreOwnCluster, _cluster, cell, absPos, _data->cellMap, tempCellMap)) {
            result = true;
            break;
        }
    }
}

__inline__ __device__ void ConstructorFunction::isObstaclePresent_rotationAndCreation(
    bool ignoreOwnCluster,
    float2 const& relPosOfNewCell,
    float2 const& centerOfRotation,
    RotationMatrices const& rotationMatrices,
    float2 const& displacementOfConstructionSite,
    bool& result)
{
    __shared__ HashMap<int2, CellAndNewAbsPos> tempCellMap;
    tempCellMap.init_blockCall(_cluster->numCellPointers * 2, _data->arrays);
    __syncthreads();

    __shared__ float2 newCenter;
    if (0 == threadIdx.x) {
        newCenter = relPosOfNewCell;
        result = false;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        auto relPos =
            getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, displacementOfConstructionSite);
        atomicAdd_block(&newCenter.x, relPos.x);
        atomicAdd_block(&newCenter.y, relPos.y);
    }
    __syncthreads();

    __shared__ Math::Matrix clusterMatrix;
    if (0 == threadIdx.x) {
        newCenter = newCenter / (_cluster->numCellPointers + 1);
        Math::rotationMatrix(_cluster->angle, clusterMatrix);
    }
    __syncthreads();

    if (!ignoreOwnCluster) {
        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto const& cell = _cluster->cellPointers[cellIndex];
            auto relPos =
                getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, displacementOfConstructionSite);
            relPos = relPos - newCenter;
            auto const absPos = _cluster->pos + Math::applyMatrix(relPos, clusterMatrix);
            tempCellMap.insertOrAssign(toInt2(absPos), CellAndNewAbsPos{ cell, absPos });
        }
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        auto relPos =
            getTransformedCellRelPos(cell, centerOfRotation, rotationMatrices, displacementOfConstructionSite);
        relPos = relPos - newCenter;
        auto const absPos = _cluster->pos + Math::applyMatrix(relPos, clusterMatrix);
        if (isObstaclePresent_helper(ignoreOwnCluster, _cluster, cell, absPos, _data->cellMap, tempCellMap)) {
            result = true;
            break;
        }
    }
}

__inline__ __device__ void ConstructorFunction::isObstaclePresent_firstCreation(
    bool ignoreOwnCluster,
    Cluster* cluster,
    float2 const& relPosOfNewCell,
    Map<Cell> const& map,
    bool& result)
{
    __shared__ HashMap<int2, CellAndNewAbsPos> tempCellMap;
    tempCellMap.init_blockCall(_cluster->numCellPointers * 2, _data->arrays);
    __syncthreads();

    __shared__ float2 newCenter;
    if (0 == threadIdx.x) {
        newCenter = relPosOfNewCell;
        result = false;
    }
    __syncthreads();

    for (auto cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        atomicAdd_block(&newCenter.x, cell->relPos.x);
        atomicAdd_block(&newCenter.y, cell->relPos.y);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        newCenter = newCenter / (cluster->numCellPointers + 1);
    }
    __syncthreads();

    Math::Matrix clusterMatrix;
    Math::rotationMatrix(cluster->angle, clusterMatrix);
    if (!ignoreOwnCluster) {
        for (auto cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];
            auto const relPos = cell->relPos - newCenter;
            auto const absPos = cluster->pos + Math::applyMatrix(relPos, clusterMatrix);
            tempCellMap.insertOrAssign(toInt2(absPos), CellAndNewAbsPos{ cell, absPos });
        }
    }
    __syncthreads();

    for (auto cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];
        auto const relPos = cell->relPos - newCenter;
        auto const absPos = cluster->pos + Math::applyMatrix(relPos, clusterMatrix);
        if (isObstaclePresent_helper(ignoreOwnCluster, cluster, cell, absPos, map, tempCellMap)) {
            result = true;
            break;
        }
    }
    __syncthreads();

    //check obstacle for cell to be constructed
    if (0 == threadIdx.x) {
        auto const absPosForNewCell = cluster->pos + Math::applyMatrix(relPosOfNewCell - newCenter, clusterMatrix);
        if (isObstaclePresent_helper(ignoreOwnCluster, cluster, nullptr, absPosForNewCell, map, tempCellMap)) {
            result = true;
        }
    }
}

__inline__ __device__ bool ConstructorFunction::isObstaclePresent_helper(
    bool ignoreOwnCluster,
    Cluster* cluster,
    Cell* cell,
    float2 const& absPos,
    Map<Cell> const& map,
    HashMap<int2, CellAndNewAbsPos>& tempMap)
{
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            float2 const lookupPos = {absPos.x + dx, absPos.y + dy};
            if (auto const otherCell = map.get(lookupPos)) {
                if (cluster != otherCell->cluster) {
                    if (map.mapDistance(otherCell->absPos, absPos) < cudaSimulationParameters.cellMinDistance) {
                        return true;
                    }

                    //check also connected cells
                    for (int i = 0; i < otherCell->numConnections; ++i) {
                        auto const connectedOtherCell = otherCell->connections[i];
                        if (map.mapDistance(connectedOtherCell->absPos, absPos)
                            < cudaSimulationParameters.cellMinDistance) {
                            return true;
                        }
                    }
                }
            }
            if (!ignoreOwnCluster) {
                auto const lookupPosInt = toInt2(lookupPos);
                if (tempMap.contains(lookupPosInt)) {
                    auto const otherCellAndNewPos = tempMap.at(lookupPosInt);

                    if (cell != otherCellAndNewPos.cell) {
                        if (map.mapDistance(otherCellAndNewPos.newAbsPos, absPos)
                            < cudaSimulationParameters.cellMinDistance) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

__inline__ __device__ void
ConstructorFunction::constructNewCell(float2 const& relPosOfNewCell, float const energyOfNewCell, Cell*& result)
{
    if (0 == threadIdx.x) {
        EntityFactory factory;
        factory.init(_data);
        result = factory.createCell(_cluster);
        result->energy = energyOfNewCell;
        result->relPos = relPosOfNewCell;
        float rotMatrix[2][2];
        Math::rotationMatrix(_cluster->angle, rotMatrix);
        result->absPos = Math::applyMatrix(result->relPos, rotMatrix) + _cluster->pos;
        result->maxConnections = _token->memory[Enums::Constr::IN_CELL_MAX_CONNECTIONS];
        result->numConnections = 0;
        result->branchNumber =
            _token->memory[Enums::Constr::IN_CELL_BRANCH_NO] % cudaSimulationParameters.cellMaxTokenBranchNumber;
        result->tokenBlocked = true;
        result->cellFunctionType = _token->memory[Enums::Constr::IN_CELL_FUNCTION];
        result->numStaticBytes = _token->memory[Enums::Constr::IN_CELL_FUNCTION_DATA];
    }
    __syncthreads();

    auto const staticDataBlock = calcPartition(result->numStaticBytes, threadIdx.x, blockDim.x);
    for (int i = staticDataBlock.startIndex; i <= staticDataBlock.endIndex; ++i) {
        result->staticData[i] = _token->memory[Enums::Constr::IN_CELL_FUNCTION_DATA + i + 1];
    }
    int offset = result->numStaticBytes + 1;
    result->numMutableBytes = _token->memory[Enums::Constr::IN_CELL_FUNCTION_DATA + offset];
    auto const mutableDataBlock = calcPartition(result->numMutableBytes, threadIdx.x, blockDim.x);
    for (int i = mutableDataBlock.startIndex; i <= mutableDataBlock.endIndex; ++i) {
        result->mutableData[i] = _token->memory[Enums::Constr::IN_CELL_FUNCTION_DATA + offset + i + 1];
    }
}

__inline__ __device__ void ConstructorFunction::constructNewToken(
    Cell* cellOfNewToken,
    Cell* sourceCellOfNewToken,
    float energyOfNewToken,
    bool duplicate,
    Token*& result)
{
    if (0 == threadIdx.x) {
        EntityFactory factory;
        factory.init(_data);

        result = factory.createToken(cellOfNewToken);
        result->sourceCell = sourceCellOfNewToken;
        result->energy = energyOfNewToken;
    }
    __syncthreads();

    auto const threadBlock = calcPartition(MAX_TOKEN_MEM_SIZE, threadIdx.x, blockDim.x);
    if (duplicate) {
        for (int i = max(1, threadBlock.startIndex); i <= threadBlock.endIndex;
             ++i) {  //do not copy branchnumber (at address 0)
            result->memory[i] = _token->memory[i];
        }
    } else {
        for (int i = max(1, threadBlock.startIndex); i <= threadBlock.endIndex;
             ++i) {  //do not copy branchnumber (at address 0)
            result->memory[i] = 0;
        }
    }
}

__inline__ __device__ void
ConstructorFunction::addCellToCluster(Cell* newCell, Cell** newCellPointers)
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        newCellPointers[cellIndex] = _cluster->cellPointers[cellIndex];
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        newCellPointers[_cluster->numCellPointers] = newCell;
        _cluster->cellPointers = newCellPointers;
        ++_cluster->numCellPointers;
    }
}

__inline__ __device__ void
ConstructorFunction::addTokenToCluster(Token* token, Token** newTokenPointers)
{
    auto const tokenBlock = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
    for (int i = tokenBlock.startIndex; i <= tokenBlock.endIndex; ++i) {
        newTokenPointers[i] = _cluster->tokenPointers[i];
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        newTokenPointers[_cluster->numTokenPointers] = token;
        _cluster->tokenPointers = newTokenPointers;
        ++_cluster->numTokenPointers;
    }
}

__inline__ __device__ void ConstructorFunction::separateConstructionWhenFinished(Cell* newCell)
{
    auto const& cell = _token->cell;
    auto const option = _token->memory[Enums::Constr::IN_OPTION] % Enums::ConstrInOption::_COUNTER;

    if (Enums::ConstrInOption::FINISH_NO_SEP == option || Enums::ConstrInOption::FINISH_WITH_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option) {
        newCell->tokenBlocked = false;
    }

    if (Enums::ConstrInOption::FINISH_WITH_SEP == option || Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option) {
        atomicExch(&_cluster->decompositionRequired, 1);
        removeConnection(newCell, cell);
    }
    if (Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option) {
        --newCell->maxConnections;
        --cell->maxConnections;
    }
}

__inline__ __device__ void ConstructorFunction::connectNewCell(
    Cell* newCell,
    Cell* cellOfConstructionSite)
{
    __shared__ AdaptMaxConnections adaptMaxConnections;
    if (0 == threadIdx.x) {
        Cell* cellOfConstructor = _token->cell;

        adaptMaxConnections = isAdaptMaxConnections(_token);

        removeConnection(cellOfConstructionSite, cellOfConstructor);
        establishConnection(newCell, cellOfConstructionSite, adaptMaxConnections);
        establishConnection(newCell, cellOfConstructor, adaptMaxConnections);
    }
    __syncthreads();

    if (newCell->numConnections >= cudaSimulationParameters.cellMaxBonds) {
        return;
    }

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        if (ClusterComponent::ConstructionSite != cell->tag) {
            continue;
        }
        if (cell == cellOfConstructionSite) {
            continue;
        }
        if (_data->cellMap.mapDistance(cell->absPos, newCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
            continue;
        }
        int origNumConnectionsOfNewCell = atomicAdd_block(&newCell->numConnections, 1);
        if (!isConnectable(cell->numConnections, cell->maxConnections, adaptMaxConnections)
            || !isConnectable(origNumConnectionsOfNewCell, newCell->maxConnections, adaptMaxConnections)) {
            atomicAdd_block(&newCell->numConnections, -1);
            continue;
        }

        newCell->connections[origNumConnectionsOfNewCell] = cell;
        cell->connections[cell->numConnections++] = newCell;

        if (adaptMaxConnections == AdaptMaxConnections::Yes) {
            newCell->maxConnections = newCell->numConnections;
            cell->maxConnections = cell->numConnections;
        }
    }
}

__inline__ __device__ void ConstructorFunction::removeConnection(Cell* cell1, Cell* cell2)
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

    remove(cell1, cell2);
    remove(cell2, cell1);
}

__inline__ __device__ auto ConstructorFunction::isAdaptMaxConnections(Token* token) -> AdaptMaxConnections
{
    return 0 == token->memory[Enums::Constr::IN_CELL_MAX_CONNECTIONS] ? AdaptMaxConnections::Yes
                                                                      : AdaptMaxConnections::No;
}

__inline__ __device__ bool
ConstructorFunction::isConnectable(int numConnections, int maxConnections, AdaptMaxConnections adaptMaxConnections)
{
    if (AdaptMaxConnections::Yes == adaptMaxConnections) {
        if (numConnections == cudaSimulationParameters.cellMaxBonds) {
            return false;
        }
    }
    if (AdaptMaxConnections::No == adaptMaxConnections) {
        if (numConnections == maxConnections) {
            return false;
        }
    }
    return true;
}

__inline__ __device__ void
ConstructorFunction::establishConnection(Cell* cell1, Cell* cell2, AdaptMaxConnections adaptMaxConnections)
{
    cell1->connections[cell1->numConnections++] = cell2;
    cell2->connections[cell2->numConnections++] = cell1;

    if (adaptMaxConnections == AdaptMaxConnections::Yes) {
        cell1->maxConnections = cell1->numConnections;
        cell2->maxConnections = cell2->numConnections;
    }
}
