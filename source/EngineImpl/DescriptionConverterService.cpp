#include "DescriptionConverterService.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include <Base/Exceptions.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/NumberGenerator.h>

#include <EngineGpuKernels/TOProvider.cuh>


namespace
{
    template <typename T>
    T* getFromHeap(uint8_t* heap, uint64_t sourceIndex)
    {
        return reinterpret_cast<T*>(&heap[sourceIndex]);
    }

    void stringToChar64(std::string const& source, Char64& target)
    {
        size_t length = std::min(source.length(), size_t(63));  // Leave space for null terminator
        std::memcpy(target, source.c_str(), length);
        target[length] = '\0';  // Ensure null termination
        // Clear remaining bytes
        for (size_t i = length + 1; i < 64; ++i) {
            target[i] = '\0';
        }
    }

    std::string char64ToString(Char64 const& source)
    {
        return std::string(source, strnlen(source, 64));
    }

    NeuralNetworkGenomeDescription convert(NeuralNetworkGenomeTO const& neuralNetworkGenomeTO)
    {
        NeuralNetworkGenomeDescription result;
        for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
            result._weights[i] = neuralNetworkGenomeTO.weights[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result._biases[i] = neuralNetworkGenomeTO.biases[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result._activationFunctions[i] = neuralNetworkGenomeTO.activationFunctions[i];
        }
        return result;
    }

    NeuralNetworkDescription convert(NeuralNetworkTO const& neuralNetworkTO)
    {
        NeuralNetworkDescription result;
        for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
            result._weights[i] = neuralNetworkTO.weights[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result._biases[i] = neuralNetworkTO.biases[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result._activationFunctions[i] = neuralNetworkTO.activationFunctions[i];
        }
        return result;
    }

    NeuralNetworkGenomeTO convert(NeuralNetworkGenomeDescription const& neuralNetworkDesc)
    {
        NeuralNetworkGenomeTO result;
        for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
            result.weights[i] = neuralNetworkDesc._weights[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result.biases[i] = neuralNetworkDesc._biases[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result.activationFunctions[i] = neuralNetworkDesc._activationFunctions[i];
        }
        return result;
    }

    NeuralNetworkTO convert(NeuralNetworkDescription const& neuralNetworkDesc)
    {
        NeuralNetworkTO result;
        for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
            result.weights[i] = neuralNetworkDesc._weights[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result.biases[i] = neuralNetworkDesc._biases[i];
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result.activationFunctions[i] = neuralNetworkDesc._activationFunctions[i];
        }
        return result;
    }


}

Description DescriptionConverterService::convertTOtoDescription(TO const& to) const
{
    Description result;

    // Genomes
    for (int i = 0; i < *to.numGenomes; ++i) {
        auto genome = createGenomeDescription(to, i);
        result._genomes.emplace_back(genome);
    }

    // Creatures
    std::unordered_map<uint64_t, uint64_t> indexByCreatureId;
    for (int i = 0; i < *to.numCreatures; ++i) {
        auto creature = createCreatureDescription(to, i);
        auto genomeTOIndex = to.creatures[i].genomeArrayIndex;
        auto genomeId = to.genomes[genomeTOIndex].id;
        creature._genomeId = genomeId;

        indexByCreatureId.emplace(creature._id, i);
        result._creatures.emplace_back(creature);
    }

    // Cells
    for (int i = 0; i < *to.numCells; ++i) {
        auto cell = createCellDescription(to, i);

        if (to.cells[i].belongToCreature) {
            auto creatureTOIndex = to.cells[i].creatureIndex;
            auto creatureId = to.creatures[creatureTOIndex].id;
            auto creatureIndex = indexByCreatureId.at(creatureId);
            auto& creature = result._creatures.at(creatureIndex);
            creature._cells.emplace_back(cell);
        } else {
            result._cells.emplace_back(cell);
        }
    }

    // Particles
    for (int i = 0; i < *to.numParticles; ++i) {
        result._particles.emplace_back(createParticleDescription(to, i));
    }

    return result;
}

TO DescriptionConverterService::convertDescriptionToTO(Description const& description) const
{
    std::vector<GenomeTO> genomeTOs;
    std::vector<CreatureTO> creatureTOs;
    std::vector<GeneTO> geneTOs;
    std::vector<NodeTO> nodeTOs;
    std::vector<CellTO> cellTOs;
    std::vector<ParticleTO> particleTOs;
    std::vector<uint8_t> heap;

    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    for (auto const& genome : description._genomes) {
        convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, genome, genomeTOIndexById);
    }

    std::unordered_map<uint64_t, uint64_t> creatureTOIndexById;
    for (auto const& creature : description._creatures) {
        convertCreatureToTO(creatureTOs, creature, genomeTOIndexById, creatureTOIndexById);
    }

    std::unordered_map<uint64_t, uint64_t> cellIndexTOById;
    for (auto const& cell : description._cells) {
        convertCellToTO(cellTOs, heap, cellIndexTOById, cell, std::nullopt, creatureTOIndexById);
    }
    for (auto const& creature : description._creatures) {
        for (auto const& cell : creature._cells) {
            convertCellToTO(cellTOs, heap, cellIndexTOById, cell, creature._id, creatureTOIndexById);
        }
    }
    description.forEachCell([&](auto const& cell) { setConnections(cellTOs, cell, cellIndexTOById); });
    for (auto const& particle : description._particles) {
        addParticle(particleTOs, particle);
    }

    return provideDataTO(creatureTOs, genomeTOs, geneTOs, nodeTOs, cellTOs, particleTOs, heap);
}

TO DescriptionConverterService::convertDescriptionToTO(CellDescription const& cell) const
{
    std::vector<CellTO> cellTOs;
    std::vector<uint8_t> heap;

    std::unordered_map<uint64_t, uint64_t> cellIndexTOById;
    std::unordered_map<uint64_t, uint64_t> creatureTOIndexById;
    convertCellToTO(cellTOs, heap, cellIndexTOById, cell, std::nullopt, creatureTOIndexById);

    return provideDataTO({}, {}, {}, {}, cellTOs, {}, heap);
}

TO DescriptionConverterService::convertDescriptionToTO(ParticleDescription const& particle) const
{
    std::vector<ParticleTO> particleTOs;
    std::vector<uint8_t> heap;
    addParticle(particleTOs, particle);

    return provideDataTO({}, {}, {}, {}, {}, particleTOs, heap);
}

TO DescriptionConverterService::convertDescriptionToTO(uint64_t creatureId, GenomeDescription const& genome) const
{
    std::vector<GenomeTO> genomeTOs;
    std::vector<CreatureTO> creatureTOs;
    std::vector<GeneTO> geneTOs;
    std::vector<NodeTO> nodeTOs;
    std::vector<CellTO> cellTOs;
    std::vector<ParticleTO> particleTOs;
    std::vector<uint8_t> heap;

    auto wrapper = CreatureDescription().id(creatureId).genomeId(genome._id);

    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, genome, genomeTOIndexById);

    std::unordered_map<uint64_t, uint64_t> creatureTOIndexById;
    convertCreatureToTO(creatureTOs, wrapper, genomeTOIndexById, creatureTOIndexById);

    return provideDataTO(creatureTOs, genomeTOs, geneTOs, nodeTOs, {}, {}, heap);
}

DescriptionConverterService::DescriptionConverterService()
{
    _collectionTOProvider = std::make_shared<_TOProvider>();
}

CellDescription DescriptionConverterService::createCellDescription(TO const& to, int cellIndex) const
{
    CellDescription result(false);

    auto const& cellTO = to.cells[cellIndex];
    result._id = cellTO.id;
    NumberGenerator::get().adaptMaxIds({.entityId = cellTO.id});
    result._pos = RealVector2D(cellTO.pos.x, cellTO.pos.y);
    result._vel = RealVector2D(cellTO.vel.x, cellTO.vel.y);
    result._energy = cellTO.energy;
    result._stiffness = cellTO.stiffness;
    std::vector<ConnectionDescription> connections;
    for (int i = 0; i < cellTO.numConnections; ++i) {
        auto const& connectionTO = cellTO.connections[i];
        ConnectionDescription connection;
        if (connectionTO.cellIndex != VALUE_NOT_SET_UINT64) {
            connection._cellId = to.cells[connectionTO.cellIndex].id;
        } else {
            connections.clear();
            break;
        }
        connection._distance = connectionTO.distance;
        connection._angleFromPrevious = connectionTO.angleFromPrevious;
        connections.emplace_back(connection);
    }
    result._connections = connections;
    result._cellState = cellTO.cellState;
    result._fixed = cellTO.fixed;
    result._sticky = cellTO.sticky;
    result._age = cellTO.age;
    result._color = cellTO.color;
    result._frontAngle = cellTO.frontAngle != VALUE_NOT_SET_FLOAT ? std::make_optional(cellTO.frontAngle) : std::nullopt;
    result._detectedByCreatureId = cellTO.detectedByCreatureId;
    result._cellTriggered = cellTO.cellTriggered;
    result._nodeIndex = cellTO.nodeIndex;
    result._parentNodeIndex = cellTO.parentNodeIndex;
    result._geneIndex = cellTO.geneIndex;
    result._frontAngleId = cellTO.frontAngleId;
    result._headCell = cellTO.headCell;

    switch (cellTO.cellType) {
    case CellType_Structure: {
        StructureCellDescription base;
        result._cellType = base;
    } break;
    case CellType_Free: {
        FreeCellDescription base;
        result._cellType = base;
    } break;
    case CellType_Base: {
        BaseDescription base;
        result._cellType = base;
    } break;
    case CellType_Depot: {
        DepotDescription transmitter;
        transmitter._mode = cellTO.cellTypeData.depot.mode;
        result._cellType = transmitter;
    } break;
    case CellType_Constructor: {
        ConstructorDescription constructor;
        constructor._autoTriggerInterval =
            cellTO.cellTypeData.constructor.autoTriggerInterval > 0 ? std::make_optional(cellTO.cellTypeData.constructor.autoTriggerInterval) : std::nullopt;
        constructor._constructionActivationTime = cellTO.cellTypeData.constructor.constructionActivationTime;
        constructor._constructionAngle = cellTO.cellTypeData.constructor.constructionAngle;
        constructor._provideEnergy = cellTO.cellTypeData.constructor.provideEnergy;
        constructor._geneIndex = cellTO.cellTypeData.constructor.geneIndex;
        constructor._lastConstructedCellId = cellTO.cellTypeData.constructor.lastConstructedCellId != VALUE_NOT_SET_UINT64
            ? std::make_optional(cellTO.cellTypeData.constructor.lastConstructedCellId)
            : std::nullopt;
        constructor._currentNodeIndex = cellTO.cellTypeData.constructor.currentNodeIndex;
        constructor._currentConcatenation = cellTO.cellTypeData.constructor.currentConcatenation;
        constructor._currentBranch = cellTO.cellTypeData.constructor.currentBranch;
        result._cellType = constructor;
    } break;
    case CellType_Sensor: {
        SensorDescription sensor;
        sensor._autoTriggerInterval =
            cellTO.cellTypeData.sensor.autoTriggerInterval > 0 ? std::make_optional(cellTO.cellTypeData.sensor.autoTriggerInterval) : std::nullopt;
        sensor._minRange = cellTO.cellTypeData.sensor.minRange;
        sensor._maxRange = cellTO.cellTypeData.sensor.maxRange;

        if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
            DetectEnergyDescription detectEnergy;
            detectEnergy._minDensity = cellTO.cellTypeData.sensor.modeData.detectEnergy.minDensity;
            sensor._mode = detectEnergy;
        } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
            DetectStructureDescription detectStructure;
            sensor._mode = detectStructure;
        } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
            DetectFreeCellDescription detectFreeCell;
            detectFreeCell._minDensity = cellTO.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
            detectFreeCell._restrictToColor = cellTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor))
                : std::nullopt;
            sensor._mode = detectFreeCell;
        } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
            DetectCreatureDescription detectCreature;
            detectCreature._minNumCells = cellTO.cellTypeData.sensor.modeData.detectCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.sensor.modeData.detectCreature.minNumCells))
                : std::nullopt;
            detectCreature._maxNumCells = cellTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.sensor.modeData.detectCreature.maxNumCells))
                : std::nullopt;
            detectCreature._restrictToColor = cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToColor != 255
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToColor))
                : std::nullopt;
            detectCreature._restrictToLineage = cellTO.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
            detectCreature._lastMatchPos = cellTO.cellTypeData.sensor.modeData.detectCreature.lastMatchPos.x != VALUE_NOT_SET_FLOAT
                ? std::make_optional(RealVector2D{cellTO.cellTypeData.sensor.modeData.detectCreature.lastMatchPos.x, cellTO.cellTypeData.sensor.modeData.detectCreature.lastMatchPos.y})
                : std::nullopt;
            sensor._mode = detectCreature;
        }

        result._cellType = sensor;
    } break;
    case CellType_Generator: {
        GeneratorDescription generator;
        generator._autoTriggerInterval = cellTO.cellTypeData.generator.autoTriggerInterval;
        generator._pulseType = cellTO.cellTypeData.generator.pulseType;
        generator._alternationInterval = cellTO.cellTypeData.generator.alternationInterval;
        generator._numPulses = cellTO.cellTypeData.generator.numPulses;
        result._cellType = generator;
    } break;
    case CellType_Attacker: {
        AttackerDescription attacker;
        result._cellType = attacker;
    } break;
    case CellType_Injector: {
        InjectorDescription injector;
        injector._mode = cellTO.cellTypeData.injector.mode;
        injector._counter = cellTO.cellTypeData.injector.counter;
        result._cellType = injector;
    } break;
    case CellType_Muscle: {
        MuscleDescription muscle;
        if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
            AutoBendingDescription bending;
            bending._maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
            bending._forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
            bending._initialAngle = cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle != VALUE_NOT_SET_FLOAT
                ? std::make_optional(cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle)
                : std::nullopt;
            bending._forward = cellTO.cellTypeData.muscle.modeData.autoBending.forward;
            bending._activation = cellTO.cellTypeData.muscle.modeData.autoBending.activation;
            bending._activationCountdown = cellTO.cellTypeData.muscle.modeData.autoBending.activationCountdown;
            bending._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
            muscle._mode = bending;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
            ManualBendingDescription bending;
            bending._maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
            bending._forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
            bending._initialAngle = cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle != VALUE_NOT_SET_FLOAT
                ? std::make_optional(cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle)
                : std::nullopt;
            bending._lastAngleDelta = cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
            bending._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
            muscle._mode = bending;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
            AngleBendingDescription bending;
            bending._maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
            bending._attractionRepulsionRatio = cellTO.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
            bending._initialAngle = cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle != VALUE_NOT_SET_FLOAT
                ? std::make_optional(cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle)
                : std::nullopt;
            muscle._mode = bending;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
            AutoCrawlingDescription crawling;
            crawling._maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
            crawling._forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
            crawling._initialDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance != VALUE_NOT_SET_FLOAT
                ? std::make_optional(cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance)
                : std::nullopt;
            crawling._lastActualDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
            crawling._forward = cellTO.cellTypeData.muscle.modeData.autoCrawling.forward;
            crawling._activation = cellTO.cellTypeData.muscle.modeData.autoCrawling.activation;
            crawling._activationCountdown = cellTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
            crawling._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
            muscle._mode = crawling;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
            ManualCrawlingDescription crawling;
            crawling._maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
            crawling._forwardBackwardRatio = cellTO.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
            crawling._initialDistance = cellTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance != VALUE_NOT_SET_FLOAT
                ? std::make_optional(cellTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance)
                : std::nullopt;
            crawling._lastActualDistance = cellTO.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
            crawling._lastDistanceDelta = cellTO.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
            crawling._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied;
            muscle._mode = crawling;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
            DirectMovementDescription movement;
            muscle._mode = movement;
        }

        muscle._lastMovementX = cellTO.cellTypeData.muscle.lastMovementX;
        muscle._lastMovementY = cellTO.cellTypeData.muscle.lastMovementY;
        result._cellType = muscle;
    } break;
    case CellType_Defender: {
        DefenderDescription defender;
        defender._mode = cellTO.cellTypeData.defender.mode;
        result._cellType = defender;
    } break;
    case CellType_Reconnector: {
        ReconnectorDescription reconnector;
        reconnector._restrictToColor =
            cellTO.cellTypeData.reconnector.restrictToColor != 255 ? std::make_optional(cellTO.cellTypeData.reconnector.restrictToColor) : std::nullopt;
        reconnector._restrictToCreatures = cellTO.cellTypeData.reconnector.restrictToCreatures;
        result._cellType = reconnector;
    } break;
    case CellType_Detonator: {
        DetonatorDescription detonator;
        detonator._state = cellTO.cellTypeData.detonator.state;
        detonator._countdown = cellTO.cellTypeData.detonator.countdown;
        result._cellType = detonator;
    } break;
    }
    if (cellTO.neuralNetworkDataIndex != VALUE_NOT_SET_UINT64) {
        auto const& neuralNetworkTO = getFromHeap<NeuralNetworkTO>(to.heap, cellTO.neuralNetworkDataIndex);
        result._neuralNetwork = convert(*neuralNetworkTO);
    }

    SignalRestrictionDescription routingRestriction;
    routingRestriction._active = cellTO.signalRestriction.active;
    routingRestriction._baseAngle = cellTO.signalRestriction.baseAngle;
    routingRestriction._openingAngle = cellTO.signalRestriction.openingAngle;
    result._signalRestriction = routingRestriction;
    result._signalState = cellTO.signalState;
    if (cellTO.signal.active) {
        SignalDescription signal;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            signal._channels[i] = cellTO.signal.channels[i];
        }
        result._signal = signal;
    }
    result._activationTime = cellTO.activationTime;
    return result;
}


NodeDescription DescriptionConverterService::createNodeDescription(NodeTO const* nodeTO) const
{
    NodeDescription nodeDesc;
    nodeDesc._referenceAngle = nodeTO->referenceAngle;
    nodeDesc._color = nodeTO->color;
    nodeDesc._numAdditionalConnections = nodeTO->numAdditionalConnections;

    nodeDesc._neuralNetwork = convert(nodeTO->neuralNetwork);
    nodeDesc._signalRestriction._active = nodeTO->signalRestriction.active;
    nodeDesc._signalRestriction._baseAngle = nodeTO->signalRestriction.baseAngle;
    nodeDesc._signalRestriction._openingAngle = nodeTO->signalRestriction.openingAngle;

    switch (nodeTO->cellType) {
    case CellTypeGenome_Base: {
        BaseGenomeDescription baseDesc;
        nodeDesc._cellType = baseDesc;
    } break;
    case CellTypeGenome_Depot: {
        DepotGenomeDescription depotDesc;
        depotDesc._mode = nodeTO->cellTypeData.depot.mode;
        nodeDesc._cellType = depotDesc;
    } break;
    case CellTypeGenome_Constructor: {
        ConstructorGenomeDescription constructorDesc;
        constructorDesc._autoTriggerInterval =
            nodeTO->cellTypeData.constructor.autoTriggerInterval > 0 ? std::make_optional(nodeTO->cellTypeData.constructor.autoTriggerInterval) : std::nullopt;
        constructorDesc._geneIndex = nodeTO->cellTypeData.constructor.geneIndex;
        constructorDesc._constructionActivationTime = nodeTO->cellTypeData.constructor.constructionActivationTime;
        constructorDesc._constructionAngle = nodeTO->cellTypeData.constructor.constructionAngle;
        constructorDesc._provideEnergy = nodeTO->cellTypeData.constructor.provideEnergy;
        nodeDesc._cellType = constructorDesc;
    } break;
    case CellTypeGenome_Sensor: {
        SensorGenomeDescription sensorDesc;
        sensorDesc._autoTriggerInterval =
            nodeTO->cellTypeData.sensor.autoTriggerInterval > 0 ? std::make_optional(nodeTO->cellTypeData.sensor.autoTriggerInterval) : std::nullopt;
        sensorDesc._minRange = nodeTO->cellTypeData.sensor.minRange;
        sensorDesc._maxRange = nodeTO->cellTypeData.sensor.maxRange;

        if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
            DetectEnergyGenomeDescription detectEnergy;
            detectEnergy._minDensity = nodeTO->cellTypeData.sensor.modeData.detectEnergy.minDensity;
            sensorDesc._mode = detectEnergy;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectStructure) {
            DetectStructureGenomeDescription detectStructure;
            sensorDesc._mode = detectStructure;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
            DetectFreeCellGenomeDescription detectFreeCell;
            detectFreeCell._minDensity = nodeTO->cellTypeData.sensor.modeData.detectFreeCell.minDensity;
            detectFreeCell._restrictToColor = nodeTO->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor))
                : std::nullopt;
            sensorDesc._mode = detectFreeCell;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectCreature) {
            DetectCreatureGenomeDescription detectCreature;
            detectCreature._minNumCells = nodeTO->cellTypeData.sensor.modeData.detectCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectCreature.minNumCells))
                : std::nullopt;
            detectCreature._maxNumCells = nodeTO->cellTypeData.sensor.modeData.detectCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectCreature.maxNumCells))
                : std::nullopt;
            detectCreature._restrictToColor = nodeTO->cellTypeData.sensor.modeData.detectCreature.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectCreature.restrictToColor))
                : std::nullopt;
            detectCreature._restrictToLineage = nodeTO->cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
            sensorDesc._mode = detectCreature;
        }

        nodeDesc._cellType = sensorDesc;
    } break;
    case CellTypeGenome_Generator: {
        GeneratorGenomeDescription generatorDesc;
        generatorDesc._autoTriggerInterval = nodeTO->cellTypeData.generator.autoTriggerInterval;
        generatorDesc._pulseType = nodeTO->cellTypeData.generator.pulseType;
        generatorDesc._alternationInterval = nodeTO->cellTypeData.generator.alternationInterval;
        nodeDesc._cellType = generatorDesc;
    } break;
    case CellTypeGenome_Attacker: {
        AttackerGenomeDescription attackerDesc;
        nodeDesc._cellType = attackerDesc;
    } break;
    case CellTypeGenome_Injector: {
        InjectorGenomeDescription injectorDesc;
        injectorDesc._mode = nodeTO->cellTypeData.injector.mode;
        nodeDesc._cellType = injectorDesc;
    } break;
    case CellTypeGenome_Muscle: {
        MuscleGenomeDescription muscleDesc;
        switch (nodeTO->cellTypeData.muscle.mode) {
        case MuscleMode_AutoBending: {
            AutoBendingGenomeDescription bendingDesc;
            bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
            bendingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
            muscleDesc._mode = bendingDesc;
        } break;
        case MuscleMode_ManualBending: {
            ManualBendingGenomeDescription bendingDesc;
            bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
            bendingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
            muscleDesc._mode = bendingDesc;
        } break;
        case MuscleMode_AngleBending: {
            AngleBendingGenomeDescription bendingDesc;
            bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
            bendingDesc._attractionRepulsionRatio = nodeTO->cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
            muscleDesc._mode = bendingDesc;
        } break;
        case MuscleMode_AutoCrawling: {
            AutoCrawlingGenomeDescription crawlingDesc;
            crawlingDesc._maxDistanceDeviation = nodeTO->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
            crawlingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
            muscleDesc._mode = crawlingDesc;
        } break;
        case MuscleMode_ManualCrawling: {
            ManualCrawlingGenomeDescription crawlingDesc;
            crawlingDesc._maxDistanceDeviation = nodeTO->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
            crawlingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
            muscleDesc._mode = crawlingDesc;
        } break;
        case MuscleMode_DirectMovement: {
            DirectMovementGenomeDescription movementDesc;
            muscleDesc._mode = movementDesc;
        } break;
        }
        nodeDesc._cellType = muscleDesc;
    } break;
    case CellTypeGenome_Defender: {
        DefenderGenomeDescription defenderDesc;
        defenderDesc._mode = nodeTO->cellTypeData.defender.mode;
        nodeDesc._cellType = defenderDesc;
    } break;
    case CellTypeGenome_Reconnector: {
        ReconnectorGenomeDescription reconnectorDesc;
        reconnectorDesc._restrictToColor = nodeTO->cellTypeData.reconnector.restrictToColor;
        reconnectorDesc._restrictToCreatures = nodeTO->cellTypeData.reconnector.restrictToCreatures;
        nodeDesc._cellType = reconnectorDesc;
    } break;
    case CellTypeGenome_Detonator: {
        DetonatorGenomeDescription detonatorDesc;
        detonatorDesc._countdown = nodeTO->cellTypeData.detonator.countdown;
        nodeDesc._cellType = detonatorDesc;
    } break;
    }
    return nodeDesc;
}

GenomeDescription DescriptionConverterService::createGenomeDescription(TO const& to, int genomeIndex) const
{
    auto const& genomeTO = to.genomes[genomeIndex];

    GenomeDescription result;
    result._id = genomeTO.id;
    NumberGenerator::get().adaptMaxIds({.entityId = genomeTO.id});
    result._name = char64ToString(genomeTO.name);
    result._frontAngle = genomeTO.frontAngle;
    result._genes.reserve(genomeTO.numGenes);

    CHECK(genomeTO.geneArrayIndex + genomeTO.numGenes <= *to.numGenes);
    for (int j = 0; j < genomeTO.numGenes; ++j) {
        auto geneTO = to.genes + genomeTO.geneArrayIndex + j;

        GeneDescription geneDesc;
        geneDesc._name = char64ToString(geneTO->name);
        geneDesc._numBranches = geneTO->numBranches;
        geneDesc._separation = geneTO->separation;
        geneDesc._shape = geneTO->shape;
        geneDesc._angleAlignment = geneTO->angleAlignment;
        geneDesc._stiffness = geneTO->stiffness;
        geneDesc._connectionDistance = geneTO->connectionDistance;
        geneDesc._numConcatenations = geneTO->numConcatenations;

        CHECK(geneTO->nodeArrayIndex + geneTO->numNodes <= *to.numNodes);
        for (int k = 0; k < geneTO->numNodes; ++k) {
            auto nodeTO = to.nodes + geneTO->nodeArrayIndex + k;
            geneDesc._nodes.emplace_back(createNodeDescription(nodeTO));
        }

        result._genes.emplace_back(geneDesc);
    }
    return result;
}

CreatureDescription DescriptionConverterService::createCreatureDescription(TO const& to, int creatureIndex) const
{
    CreatureDescription result;

    auto const& creatureTO = to.creatures[creatureIndex];
    result._id = creatureTO.id;
    NumberGenerator::get().adaptMaxIds({.entityId = creatureTO.id});
    result._ancestorId = creatureTO.ancestorId != VALUE_NOT_SET_UINT64 ? std::make_optional(creatureTO.ancestorId) : std::nullopt;
    result._generation = creatureTO.generation;
    result._lineageId = creatureTO.lineageId;
    NumberGenerator::get().adaptMaxIds({.entityId = creatureTO.lineageId});
    result._numCells = creatureTO.numCells;
    result._frontAngleId = creatureTO.frontAngleId;

    return result;
}

ParticleDescription DescriptionConverterService::createParticleDescription(TO const& to, int particleIndex) const
{
    auto const& particle = to.particles[particleIndex];
    NumberGenerator::get().adaptMaxIds({.entityId = particle.id});
    return ParticleDescription()
        .id(particle.id)
        .pos({particle.pos.x, particle.pos.y})
        .vel({particle.vel.x, particle.vel.y})
        .energy(particle.energy)
        .color(particle.color);
}

void DescriptionConverterService::convertGenomeToTO(
    std::vector<GenomeTO>& genomeTOs,
    std::vector<GeneTO>& geneTOs,
    std::vector<NodeTO>& nodeTOs,
    GenomeDescription const& genome,
    std::unordered_map<uint64_t, uint64_t>& genomeTOIndexById) const
{
    auto genomeTOIndex = genomeTOs.size();
    genomeTOs.resize(genomeTOIndex + 1);
    GenomeTO& genomeTO = genomeTOs.at(genomeTOIndex);

    genomeTOIndexById.insert_or_assign(genome._id, genomeTOIndex);

    auto geneArrayStartIndex = geneTOs.size();
    geneTOs.resize(geneArrayStartIndex + genome._genes.size());

    stringToChar64(genome._name, genomeTO.name);
    genomeTO.id = genome._id;
    genomeTO.frontAngle = genome._frontAngle;
    genomeTO.numGenes = toInt(genome._genes.size());
    genomeTO.geneArrayIndex = geneArrayStartIndex;
    genomeTO.genomeIndexOnGpu = VALUE_NOT_SET_UINT64;

    for (auto const& [geneIndex, geneDesc] : genome._genes | boost::adaptors::indexed(0)) {
        GeneTO& geneTO = geneTOs.at(geneArrayStartIndex + geneIndex);

        stringToChar64(geneDesc._name, geneTO.name);
        geneTO.shape = geneDesc._shape;
        geneTO.numBranches = static_cast<uint8_t>(geneDesc._numBranches);
        geneTO.separation = geneDesc._separation;
        geneTO.angleAlignment = geneDesc._angleAlignment;
        geneTO.stiffness = geneDesc._stiffness;
        geneTO.connectionDistance = geneDesc._connectionDistance;
        geneTO.numConcatenations = geneDesc._numConcatenations;
        geneTO.numNodes = toInt(geneDesc._nodes.size());

        auto nodeArrayStartIndex = nodeTOs.size();
        nodeTOs.resize(nodeArrayStartIndex + geneDesc._nodes.size());

        geneTO.nodeArrayIndex = nodeArrayStartIndex;
        for (auto const& [nodeIndex, nodeDesc] : geneDesc._nodes | boost::adaptors::indexed(0)) {
            NodeTO& nodeTO = nodeTOs.at(nodeArrayStartIndex + nodeIndex);
            nodeTO.referenceAngle = nodeDesc._referenceAngle;
            nodeTO.color = nodeDesc._color;
            nodeTO.numAdditionalConnections = nodeDesc._numAdditionalConnections;
            nodeTO.signalRestriction.active = nodeDesc._signalRestriction._active;
            nodeTO.signalRestriction.baseAngle = nodeDesc._signalRestriction._baseAngle;
            nodeTO.signalRestriction.openingAngle = nodeDesc._signalRestriction._openingAngle;
            nodeTO.neuralNetwork = convert(nodeDesc._neuralNetwork);

            nodeTO.cellType = nodeDesc.getCellType();
            switch (nodeDesc.getCellType()) {
            case CellTypeGenome_Base: {
            } break;
            case CellTypeGenome_Depot: {
                auto const& depotDesc = std::get<DepotGenomeDescription>(nodeDesc._cellType);
                auto& depotTO = nodeTO.cellTypeData.depot;
                depotTO.mode = depotDesc._mode;
            } break;
            case CellTypeGenome_Constructor: {
                auto const& constructorDesc = std::get<ConstructorGenomeDescription>(nodeDesc._cellType);
                auto& constructorTO = nodeTO.cellTypeData.constructor;
                constructorTO.autoTriggerInterval = static_cast<uint32_t>(constructorDesc._autoTriggerInterval.value_or(0));
                constructorTO.geneIndex = constructorDesc._geneIndex;
                constructorTO.constructionActivationTime = constructorDesc._constructionActivationTime;
                constructorTO.constructionAngle = constructorDesc._constructionAngle;
                constructorTO.provideEnergy = constructorDesc._provideEnergy;
            } break;
            case CellTypeGenome_Sensor: {
                auto const& sensorDesc = std::get<SensorGenomeDescription>(nodeDesc._cellType);
                auto& sensorTO = nodeTO.cellTypeData.sensor;
                sensorTO.autoTriggerInterval = static_cast<uint32_t>(sensorDesc._autoTriggerInterval.value_or(0));
                sensorTO.minRange = static_cast<int8_t>(sensorDesc._minRange);
                sensorTO.maxRange = static_cast<int8_t>(sensorDesc._maxRange);
                sensorTO.mode = sensorDesc.getMode();

                if (sensorTO.mode == SensorMode_DetectEnergy) {
                    auto const& detectEnergyDesc = std::get<DetectEnergyGenomeDescription>(sensorDesc._mode);
                    auto& detectEnergyTO = sensorTO.modeData.detectEnergy;
                    detectEnergyTO.minDensity = detectEnergyDesc._minDensity;
                } else if (sensorTO.mode == SensorMode_DetectStructure) {
                } else if (sensorTO.mode == SensorMode_DetectFreeCell) {
                    auto const& detectFreeCellDesc = std::get<DetectFreeCellGenomeDescription>(sensorDesc._mode);
                    auto& detectFreeCellTO = sensorTO.modeData.detectFreeCell;
                    detectFreeCellTO.minDensity = detectFreeCellDesc._minDensity;
                    detectFreeCellTO.restrictToColor = static_cast<uint8_t>(detectFreeCellDesc._restrictToColor.value_or(255));
                } else if (sensorTO.mode == SensorMode_DetectCreature) {
                    auto const& detectCreatureDesc = std::get<DetectCreatureGenomeDescription>(sensorDesc._mode);
                    auto& detectCreatureTO = sensorTO.modeData.detectCreature;
                    detectCreatureTO.minNumCells = static_cast<uint32_t>(detectCreatureDesc._minNumCells.value_or(0));
                    detectCreatureTO.maxNumCells = static_cast<uint32_t>(detectCreatureDesc._maxNumCells.value_or(0));
                    detectCreatureTO.restrictToColor = static_cast<uint8_t>(detectCreatureDesc._restrictToColor.value_or(255));
                    detectCreatureTO.restrictToLineage = detectCreatureDesc._restrictToLineage;
                }
            } break;
            case CellTypeGenome_Generator: {
                auto const& generatorDesc = std::get<GeneratorGenomeDescription>(nodeDesc._cellType);
                auto& generatorTO = nodeTO.cellTypeData.generator;
                generatorTO.autoTriggerInterval = generatorDesc._autoTriggerInterval;
                generatorTO.pulseType = generatorDesc._pulseType;
                generatorTO.alternationInterval = generatorDesc._alternationInterval;
            } break;
            case CellTypeGenome_Attacker: {
            } break;
            case CellTypeGenome_Injector: {
                auto const& injectorDesc = std::get<InjectorGenomeDescription>(nodeDesc._cellType);
                auto& injectorTO = nodeTO.cellTypeData.injector;
                injectorTO.mode = injectorDesc._mode;
            } break;
            case CellTypeGenome_Muscle: {
                auto const& muscleDesc = std::get<MuscleGenomeDescription>(nodeDesc._cellType);
                auto& muscleTO = nodeTO.cellTypeData.muscle;
                muscleTO.mode = muscleDesc.getMode();
                switch (muscleDesc.getMode()) {
                case MuscleMode_AutoBending: {
                    auto const& autoBendingDesc = std::get<AutoBendingGenomeDescription>(muscleDesc._mode);
                    auto& autoBendingTO = muscleTO.modeData.autoBending;
                    autoBendingTO.maxAngleDeviation = autoBendingDesc._maxAngleDeviation;
                    autoBendingTO.forwardBackwardRatio = autoBendingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_ManualBending: {
                    auto const& manualBendingDesc = std::get<ManualBendingGenomeDescription>(muscleDesc._mode);
                    auto& manualBendingTO = muscleTO.modeData.manualBending;
                    manualBendingTO.maxAngleDeviation = manualBendingDesc._maxAngleDeviation;
                    manualBendingTO.forwardBackwardRatio = manualBendingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_AngleBending: {
                    auto const& angleBendingDesc = std::get<AngleBendingGenomeDescription>(muscleDesc._mode);
                    auto& angleBendingTO = muscleTO.modeData.angleBending;
                    angleBendingTO.maxAngleDeviation = angleBendingDesc._maxAngleDeviation;
                    angleBendingTO.attractionRepulsionRatio = angleBendingDesc._attractionRepulsionRatio;
                } break;
                case MuscleMode_AutoCrawling: {
                    auto const& autoCrawlingDesc = std::get<AutoCrawlingGenomeDescription>(muscleDesc._mode);
                    auto& autoCrawlingTO = muscleTO.modeData.autoCrawling;
                    autoCrawlingTO.maxDistanceDeviation = autoCrawlingDesc._maxDistanceDeviation;
                    autoCrawlingTO.forwardBackwardRatio = autoCrawlingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_ManualCrawling: {
                    auto const& manualCrawlingDesc = std::get<ManualCrawlingGenomeDescription>(muscleDesc._mode);
                    auto& manualCrawlingTO = muscleTO.modeData.manualCrawling;
                    manualCrawlingTO.maxDistanceDeviation = manualCrawlingDesc._maxDistanceDeviation;
                    manualCrawlingTO.forwardBackwardRatio = manualCrawlingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_DirectMovement: {
                } break;
                }
            } break;
            case CellTypeGenome_Defender: {
                auto const& defenderDesc = std::get<DefenderGenomeDescription>(nodeDesc._cellType);
                auto& defenderTO = nodeTO.cellTypeData.defender;
                defenderTO.mode = defenderDesc._mode;
            } break;
            case CellTypeGenome_Reconnector: {
                auto const& reconnectorDesc = std::get<ReconnectorGenomeDescription>(nodeDesc._cellType);
                auto& reconnectorTO = nodeTO.cellTypeData.reconnector;
                reconnectorTO.restrictToColor = reconnectorDesc._restrictToColor.value_or(255);
                reconnectorTO.restrictToCreatures = reconnectorDesc._restrictToCreatures;
            } break;
            case CellTypeGenome_Detonator: {
                auto const& detonatorDesc = std::get<DetonatorGenomeDescription>(nodeDesc._cellType);
                auto& detonatorTO = nodeTO.cellTypeData.detonator;
                detonatorTO.countdown = detonatorDesc._countdown;
            } break;
            }
        }
    }
}

void DescriptionConverterService::convertCreatureToTO(
    std::vector<CreatureTO>& creatureTOs,
    CreatureDescription const& creatureDesc,
    std::unordered_map<uint64_t, uint64_t> const& genomeTOIndexById,
    std::unordered_map<uint64_t, uint64_t>& creatureTOIndexById) const
{
    auto creatureTOIndex = creatureTOs.size();
    creatureTOs.resize(creatureTOIndex + 1);

    CreatureTO& creatureTO = creatureTOs.at(creatureTOIndex);
    creatureTOIndexById.insert_or_assign(creatureDesc._id, creatureTOIndex);

    creatureTO.id = creatureDesc._id;
    creatureTO.ancestorId = creatureDesc._ancestorId.value_or(VALUE_NOT_SET_UINT64);
    creatureTO.generation = creatureDesc._generation;
    creatureTO.lineageId = creatureDesc._lineageId;
    creatureTO.frontAngleId = creatureDesc._frontAngleId;
    creatureTO.numCells = creatureDesc._numCells;
    creatureTO.genomeArrayIndex = genomeTOIndexById.at(creatureDesc._genomeId);
}

namespace
{
    void checkAndCorrectInvalidEnergy(float& energy)
    {
        if (std::isnan(energy) || energy < 0 || energy > 1e12) {
            energy = 0;
        }
    }
}

void DescriptionConverterService::convertCellToTO(
    std::vector<CellTO>& cellTOs,
    std::vector<uint8_t>& heap,
    std::unordered_map<uint64_t, uint64_t>& cellTOIndexById,
    CellDescription const& cellDesc,
    std::optional<uint64_t> const& creatureId,
    std::unordered_map<uint64_t, uint64_t> const& creatureTOIndexById) const
{
    auto cellIndex = cellTOs.size();
    cellTOs.resize(cellIndex + 1);

    CellTO& cellTO = cellTOs.at(cellIndex);
    cellTO.id = cellDesc._id;
    cellTOIndexById.insert_or_assign(cellTO.id, cellIndex);

    cellTO.belongToCreature = creatureId.has_value();
    if (cellTO.belongToCreature) {
        cellTO.creatureIndex = creatureTOIndexById.at(creatureId.value());
    }
    cellTO.pos = {cellDesc._pos.x, cellDesc._pos.y};
    cellTO.vel = {cellDesc._vel.x, cellDesc._vel.y};
    cellTO.energy = cellDesc._energy;
    checkAndCorrectInvalidEnergy(cellTO.energy);
    cellTO.stiffness = cellDesc._stiffness;
    cellTO.cellState = cellDesc._cellState;
    cellTO.cellType = cellDesc.getCellType();
    cellTO.detectedByCreatureId = cellDesc._detectedByCreatureId;
    cellTO.cellTriggered = cellDesc._cellTriggered;
    cellTO.nodeIndex = cellDesc._nodeIndex;
    cellTO.parentNodeIndex = cellDesc._parentNodeIndex;
    cellTO.geneIndex = cellDesc._geneIndex;
    cellTO.frontAngle = cellDesc._frontAngle.value_or(VALUE_NOT_SET_FLOAT);
    cellTO.frontAngleId = cellDesc._frontAngleId;
    cellTO.headCell = cellDesc._headCell;

    auto cellType = cellDesc.getCellType();
    if (cellDesc._neuralNetwork.has_value()) {
        cellTO.neuralNetworkDataIndex = heap.size();
        heap.resize(heap.size() + sizeof(NeuralNetworkTO));
        auto neuralNetworkTO = reinterpret_cast<NeuralNetworkTO*>(heap.data() + heap.size() - sizeof(NeuralNetworkTO));
        *neuralNetworkTO = convert(*cellDesc._neuralNetwork);
    } else {
        cellTO.neuralNetworkDataIndex = VALUE_NOT_SET_UINT64;
    }
    switch (cellType) {
    case CellType_Base: {
        BaseTO baseTO;
        cellTO.cellTypeData.base = baseTO;
    } break;
    case CellType_Depot: {
        auto const& transmitterDesc = std::get<DepotDescription>(cellDesc._cellType);
        DepotTO& transmitterTO = cellTO.cellTypeData.depot;
        transmitterTO.mode = transmitterDesc._mode;
    } break;
    case CellType_Constructor: {
        auto const& constructorDesc = std::get<ConstructorDescription>(cellDesc._cellType);
        ConstructorTO& constructorTO = cellTO.cellTypeData.constructor;
        constructorTO.autoTriggerInterval = static_cast<uint32_t>(constructorDesc._autoTriggerInterval.value_or(0));
        constructorTO.constructionActivationTime = constructorDesc._constructionActivationTime;
        constructorTO.constructionAngle = constructorDesc._constructionAngle;
        constructorTO.provideEnergy = constructorDesc._provideEnergy;
        constructorTO.geneIndex = static_cast<uint16_t>(constructorDesc._geneIndex);
        constructorTO.lastConstructedCellId = constructorDesc._lastConstructedCellId.value_or(VALUE_NOT_SET_UINT64);
        constructorTO.currentNodeIndex = static_cast<uint16_t>(constructorDesc._currentNodeIndex);
        constructorTO.currentConcatenation = static_cast<uint16_t>(constructorDesc._currentConcatenation);
        constructorTO.currentBranch = static_cast<uint8_t>(constructorDesc._currentBranch);
    } break;
    case CellType_Sensor: {
        auto const& sensorDesc = std::get<SensorDescription>(cellDesc._cellType);
        SensorTO& sensorTO = cellTO.cellTypeData.sensor;
        sensorTO.autoTriggerInterval = static_cast<uint32_t>(sensorDesc._autoTriggerInterval.value_or(0));
        sensorTO.minRange = static_cast<int8_t>(sensorDesc._minRange);
        sensorTO.maxRange = static_cast<int8_t>(sensorDesc._maxRange);
        sensorTO.mode = sensorDesc.getMode();

        if (sensorTO.mode == SensorMode_DetectEnergy) {
            auto const& detectEnergyDesc = std::get<DetectEnergyDescription>(sensorDesc._mode);
            DetectEnergyTO& detectEnergyTO = sensorTO.modeData.detectEnergy;
            detectEnergyTO.minDensity = detectEnergyDesc._minDensity;
        } else if (sensorTO.mode == SensorMode_DetectStructure) {
        } else if (sensorTO.mode == SensorMode_DetectFreeCell) {
            auto const& detectFreeCellDesc = std::get<DetectFreeCellDescription>(sensorDesc._mode);
            DetectFreeCellTO& detectFreeCellTO = sensorTO.modeData.detectFreeCell;
            detectFreeCellTO.minDensity = detectFreeCellDesc._minDensity;
            detectFreeCellTO.restrictToColor = static_cast<uint8_t>(detectFreeCellDesc._restrictToColor.value_or(255));
        } else if (sensorTO.mode == SensorMode_DetectCreature) {
            auto const& detectCreatureDesc = std::get<DetectCreatureDescription>(sensorDesc._mode);
            DetectCreatureTO& detectCreatureTO = sensorTO.modeData.detectCreature;
            detectCreatureTO.minNumCells = static_cast<uint32_t>(detectCreatureDesc._minNumCells.value_or(0));
            detectCreatureTO.maxNumCells = static_cast<uint32_t>(detectCreatureDesc._maxNumCells.value_or(0));
            detectCreatureTO.restrictToColor = static_cast<uint8_t>(detectCreatureDesc._restrictToColor.value_or(255));
            detectCreatureTO.restrictToLineage = detectCreatureDesc._restrictToLineage;
            if (detectCreatureDesc._lastMatchPos.has_value()) {
                detectCreatureTO.lastMatchPos = {detectCreatureDesc._lastMatchPos->x, detectCreatureDesc._lastMatchPos->y};
            } else {
                detectCreatureTO.lastMatchPos = {VALUE_NOT_SET_FLOAT, VALUE_NOT_SET_FLOAT};
            }
        }
    } break;
    case CellType_Generator: {
        auto const& generatorDesc = std::get<GeneratorDescription>(cellDesc._cellType);
        GeneratorTO& generatorTO = cellTO.cellTypeData.generator;
        generatorTO.autoTriggerInterval = generatorDesc._autoTriggerInterval;
        generatorTO.pulseType = generatorDesc._pulseType;
        generatorTO.alternationInterval = generatorDesc._alternationInterval;
        generatorTO.numPulses = generatorDesc._numPulses;
    } break;
    case CellType_Attacker: {
        //auto const& attackerDesc = std::get<AttackerDescription>(cellDesc._cellType);
        //AttackerTO& attackerTO = cellTO.cellTypeData.attacker;
    } break;
    case CellType_Injector: {
        auto const& injectorDesc = std::get<InjectorDescription>(cellDesc._cellType);
        InjectorTO& injectorTO = cellTO.cellTypeData.injector;
        injectorTO.mode = injectorDesc._mode;
        injectorTO.counter = injectorDesc._counter;
    } break;
    case CellType_Muscle: {
        auto const& muscleDesc = std::get<MuscleDescription>(cellDesc._cellType);
        MuscleTO& muscleTO = cellTO.cellTypeData.muscle;
        muscleTO.mode = muscleDesc.getMode();
        if (muscleTO.mode == MuscleMode_AutoBending) {
            auto const& bendingDesc = std::get<AutoBendingDescription>(muscleDesc._mode);
            AutoBendingTO& bendingTO = muscleTO.modeData.autoBending;
            bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
            bendingTO.forwardBackwardRatio = bendingDesc._forwardBackwardRatio;
            bendingTO.initialAngle = bendingDesc._initialAngle.value_or(VALUE_NOT_SET_FLOAT);
            bendingTO.forward = bendingDesc._forward;
            bendingTO.activation = bendingDesc._activation;
            bendingTO.activationCountdown = bendingDesc._activationCountdown;
            bendingTO.impulseAlreadyApplied = bendingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_ManualBending) {
            auto const& bendingDesc = std::get<ManualBendingDescription>(muscleDesc._mode);
            ManualBendingTO& bendingTO = muscleTO.modeData.manualBending;
            bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
            bendingTO.forwardBackwardRatio = bendingDesc._forwardBackwardRatio;
            bendingTO.initialAngle = bendingDesc._initialAngle.value_or(VALUE_NOT_SET_FLOAT);
            bendingTO.lastAngleDelta = bendingDesc._lastAngleDelta;
            bendingTO.impulseAlreadyApplied = bendingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_AngleBending) {
            auto const& bendingDesc = std::get<AngleBendingDescription>(muscleDesc._mode);
            AngleBendingTO& bendingTO = muscleTO.modeData.angleBending;
            bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
            bendingTO.attractionRepulsionRatio = bendingDesc._attractionRepulsionRatio;
            bendingTO.initialAngle = bendingDesc._initialAngle.value_or(VALUE_NOT_SET_FLOAT);
        } else if (muscleTO.mode == MuscleMode_AutoCrawling) {
            auto const& crawlingDesc = std::get<AutoCrawlingDescription>(muscleDesc._mode);
            AutoCrawlingTO& crawlingTO = muscleTO.modeData.autoCrawling;
            crawlingTO.maxDistanceDeviation = crawlingDesc._maxDistanceDeviation;
            crawlingTO.forwardBackwardRatio = crawlingDesc._forwardBackwardRatio;
            crawlingTO.initialDistance = crawlingDesc._initialDistance.value_or(VALUE_NOT_SET_FLOAT);
            crawlingTO.lastActualDistance = crawlingDesc._lastActualDistance;
            crawlingTO.forward = crawlingDesc._forward;
            crawlingTO.activation = crawlingDesc._activation;
            crawlingTO.activationCountdown = crawlingDesc._activationCountdown;
            crawlingTO.impulseAlreadyApplied = crawlingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_ManualCrawling) {
            auto const& crawlingDesc = std::get<ManualCrawlingDescription>(muscleDesc._mode);
            ManualCrawlingTO& crawlingTO = muscleTO.modeData.manualCrawling;
            crawlingTO.maxDistanceDeviation = crawlingDesc._maxDistanceDeviation;
            crawlingTO.forwardBackwardRatio = crawlingDesc._forwardBackwardRatio;
            crawlingTO.initialDistance = crawlingDesc._initialDistance.value_or(VALUE_NOT_SET_FLOAT);
            crawlingTO.lastActualDistance = crawlingDesc._lastActualDistance;
            crawlingTO.lastDistanceDelta = crawlingDesc._lastDistanceDelta;
            crawlingTO.impulseAlreadyApplied = crawlingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_DirectMovement) {
        }
        muscleTO.lastMovementX = muscleDesc._lastMovementX;
        muscleTO.lastMovementY = muscleDesc._lastMovementY;
    } break;
    case CellType_Defender: {
        auto const& defenderDesc = std::get<DefenderDescription>(cellDesc._cellType);
        DefenderTO& defenderTO = cellTO.cellTypeData.defender;
        defenderTO.mode = defenderDesc._mode;
    } break;
    case CellType_Reconnector: {
        auto const& reconnectorDesc = std::get<ReconnectorDescription>(cellDesc._cellType);
        ReconnectorTO& reconnectorTO = cellTO.cellTypeData.reconnector;
        reconnectorTO.restrictToColor = toUInt8(reconnectorDesc._restrictToColor.value_or(255));
        reconnectorTO.restrictToCreatures = reconnectorDesc._restrictToCreatures;
    } break;
    case CellType_Detonator: {
        auto const& detonatorDesc = std::get<DetonatorDescription>(cellDesc._cellType);
        DetonatorTO& detonatorTO = cellTO.cellTypeData.detonator;
        detonatorTO.state = detonatorDesc._state;
        detonatorTO.countdown = detonatorDesc._countdown;
    } break;
    }
    cellTO.signalRestriction.active = cellDesc._signalRestriction._active;
    cellTO.signalRestriction.baseAngle = cellDesc._signalRestriction._baseAngle;
    cellTO.signalRestriction.openingAngle = cellDesc._signalRestriction._openingAngle;
    cellTO.signalState = cellDesc._signalState;
    cellTO.signal.active = cellDesc._signal.has_value();
    if (cellTO.signal.active) {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cellDesc._signal->_channels[i];
        }
    }
    cellTO.activationTime = cellDesc._activationTime;
    cellTO.numConnections = 0;
    cellTO.fixed = cellDesc._fixed;
    cellTO.sticky = cellDesc._sticky;
    cellTO.age = cellDesc._age;
    cellTO.color = cellDesc._color;
}

void DescriptionConverterService::addParticle(std::vector<ParticleTO>& particleTOs, ParticleDescription const& particleDesc) const
{
    auto& particleTO = particleTOs.emplace_back();

    particleTO.id = particleDesc._id;
    particleTO.pos = {particleDesc._pos.x, particleDesc._pos.y};
    particleTO.vel = {particleDesc._vel.x, particleDesc._vel.y};
    particleTO.energy = particleDesc._energy;
    checkAndCorrectInvalidEnergy(particleTO.energy);
    particleTO.color = particleDesc._color;
}

void DescriptionConverterService::setConnections(
    std::vector<CellTO>& cellTOs,
    CellDescription const& cellToAdd,
    std::unordered_map<uint64_t, uint64_t> const& cellIndexByIds) const
{
    int index = 0;
    auto& cellTO = cellTOs.at(cellIndexByIds.at(cellToAdd._id));
    float angleOffset = 0;
    for (ConnectionDescription const& connection : cellToAdd._connections) {
        cellTO.connections[index].cellIndex = cellIndexByIds.at(connection._cellId);
        cellTO.connections[index].distance = connection._distance;
        cellTO.connections[index].angleFromPrevious = connection._angleFromPrevious + angleOffset;
        ++index;
        angleOffset = 0;
    }
    if (angleOffset != 0 && index > 0) {
        cellTO.connections[0].angleFromPrevious += angleOffset;
    }
    cellTO.numConnections = index;
}

TO DescriptionConverterService::provideDataTO(
    std::vector<CreatureTO> const& creatureTOs,
    std::vector<GenomeTO> const& genomeTOs,
    std::vector<GeneTO> const& geneTOs,
    std::vector<NodeTO> const& nodeTOs,
    std::vector<CellTO> const& cellTOs,
    std::vector<ParticleTO> const& particleTOs,
    std::vector<uint8_t> const& heap) const
{
    TO result = _collectionTOProvider->provideDataTO(
        {.creatures = creatureTOs.size(),
         .genomes = genomeTOs.size(),
         .genes = geneTOs.size(),
         .nodes = nodeTOs.size(),
         .cells = cellTOs.size(),
         .particles = particleTOs.size(),
         .heap = heap.size()});

    *result.numCreatures = creatureTOs.size();
    *result.numGenomes = genomeTOs.size();
    *result.numGenes = geneTOs.size();
    *result.numNodes = nodeTOs.size();
    *result.numCells = cellTOs.size();
    *result.numParticles = particleTOs.size();
    *result.heapSize = heap.size();

    std::memcpy(result.creatures, creatureTOs.data(), creatureTOs.size() * sizeof(CreatureTO));
    std::memcpy(result.genomes, genomeTOs.data(), genomeTOs.size() * sizeof(GenomeTO));
    std::memcpy(result.genes, geneTOs.data(), geneTOs.size() * sizeof(GeneTO));
    std::memcpy(result.nodes, nodeTOs.data(), nodeTOs.size() * sizeof(NodeTO));
    std::memcpy(result.cells, cellTOs.data(), cellTOs.size() * sizeof(CellTO));
    std::memcpy(result.particles, particleTOs.data(), particleTOs.size() * sizeof(ParticleTO));
    std::memcpy(result.heap, heap.data(), heap.size());

    return result;
}
