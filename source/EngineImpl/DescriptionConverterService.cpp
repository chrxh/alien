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

    // Helper function to copy memory entries from TO to Description
    template <typename TOEntry, typename DescEntry>
    void copyMemoryEntriesToDescription(std::vector<DescEntry>& target, TOEntry const* source, int numEntries)
    {
        target.resize(numEntries);
        for (int i = 0; i < numEntries; ++i) {
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                target[i]._channels[j] = source[i].channels[j];
            }
        }
    }

    // Helper function to copy memory entries from Description to TO
    template <typename DescEntry, typename TOEntry>
    void copyMemoryEntriesToTO(TOEntry* target, std::vector<DescEntry> const& source)
    {
        for (int i = 0, j = toInt(source.size()); i < j; ++i) {
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                target[i].channels[k] = source[i]._channels[k];
            }
        }
    }

    void stringToChar64(Char64& target, std::string const& source)
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
        convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, heap, genome, genomeTOIndexById);
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
    std::vector<uint8_t> heap;

    auto wrapper = CreatureDescription().id(creatureId).genomeId(genome._id);

    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, heap, genome, genomeTOIndexById);

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
    result._usableEnergy = cellTO.usableEnergy;
    result._rawEnergy = cellTO.rawEnergy;
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
        transmitter._storageLimit = cellTO.cellTypeData.depot.storageLimit;
        transmitter._storedUsableEnergy = cellTO.cellTypeData.depot.storedUsableEnergy;
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

        if (cellTO.cellTypeData.sensor.mode == SensorMode_Telemetry) {
            TelemetryDescription telemetry;
            sensor._mode = telemetry;
        } else if (cellTO.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
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
            sensor._mode = detectCreature;
        }
        if (cellTO.cellTypeData.sensor.lastMatchAvailable) {
            SensorLastMatchDescription lastMatchDesc;
            lastMatchDesc._creatureId = cellTO.cellTypeData.sensor.lastMatch.creatureId;
            lastMatchDesc._pos = RealVector2D{
                cellTO.cellTypeData.sensor.lastMatch.pos.x, cellTO.cellTypeData.sensor.lastMatch.pos.y};
            sensor._lastMatch = lastMatchDesc;
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
        if (cellTO.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
            AttackFreeCellDescription attackFreeCell;
            attackFreeCell._restrictToColor = cellTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor))
                : std::nullopt;
            attacker._mode = attackFreeCell;
        } else if (cellTO.cellTypeData.attacker.mode == AttackerMode_Creature) {
            AttackCreatureDescription attackCreature;
            attackCreature._minNumCells = cellTO.cellTypeData.attacker.modeData.attackCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.attacker.modeData.attackCreature.minNumCells))
                : std::nullopt;
            attackCreature._maxNumCells = cellTO.cellTypeData.attacker.modeData.attackCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.attacker.modeData.attackCreature.maxNumCells))
                : std::nullopt;
            attackCreature._restrictToColor = cellTO.cellTypeData.attacker.modeData.attackCreature.restrictToColor != 255
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.attacker.modeData.attackCreature.restrictToColor))
                : std::nullopt;
            attackCreature._restrictToLineage = cellTO.cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
            attacker._mode = attackCreature;
        }
        result._cellType = attacker;
    } break;
    case CellType_Injector: {
        InjectorDescription injector;
        injector._geneIndex = cellTO.cellTypeData.injector.geneIndex;
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
        if (cellTO.cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
            ReconnectStructureDescription reconnectStructure;
            reconnector._mode = reconnectStructure;
        } else if (cellTO.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
            ReconnectFreeCellDescription reconnectFreeCell;
            reconnectFreeCell._restrictToColor = cellTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor))
                : std::nullopt;
            reconnector._mode = reconnectFreeCell;
        } else if (cellTO.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
            ReconnectCreatureDescription reconnectCreature;
            reconnectCreature._minNumCells = cellTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells))
                : std::nullopt;
            reconnectCreature._maxNumCells = cellTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells))
                : std::nullopt;
            reconnectCreature._restrictToColor = cellTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor != 255
                ? std::make_optional(static_cast<int>(cellTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor))
                : std::nullopt;
            reconnectCreature._restrictToLineage = cellTO.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
            reconnector._mode = reconnectCreature;
        }
        result._cellType = reconnector;
    } break;
    case CellType_Detonator: {
        DetonatorDescription detonator;
        detonator._state = cellTO.cellTypeData.detonator.state;
        detonator._countdown = cellTO.cellTypeData.detonator.countdown;
        result._cellType = detonator;
    } break;
    case CellType_Digestor: {
        DigestorDescription digestor;
        digestor._rawEnergyConductivity = cellTO.cellTypeData.digestor.rawEnergyConductivity;
        result._cellType = digestor;
    } break;
    case CellType_Memory: {
        MemoryDescription memory;
        auto const& memoryTO = cellTO.cellTypeData.memory;
        if (memoryTO.mode == MemoryMode_SignalDelay) {
            SignalDelayDescription signalDelay;
            signalDelay._delay = memoryTO.modeData.signalDelay.delay;
            signalDelay._numSignalEntriesInitialized = memoryTO.modeData.signalDelay.numSignalEntriesInitialized;
            signalDelay._ringBufferIndex = memoryTO.modeData.signalDelay.ringBufferIndex;
            memory._mode = signalDelay;
        } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
            SignalRecorderDescription signalRecorder;
            signalRecorder._readOnly = memoryTO.modeData.signalRecorder.readOnly;
            signalRecorder._state = memoryTO.modeData.signalRecorder.state;
            signalRecorder._numWrittenSignalEntries = memoryTO.modeData.signalRecorder.numWrittenSignalEntries;
            signalRecorder._numReadSignalEntries = memoryTO.modeData.signalRecorder.numReadSignalEntries;
            memory._mode = signalRecorder;
        } else if (memoryTO.mode == MemoryMode_SignalStorage) {
            SignalStorageDescription signalStorage;
            signalStorage._readOnly = memoryTO.modeData.signalStorage.readOnly;
            memory._mode = signalStorage;
        } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
            SignalIntegratorDescription signalIntegrator;
            signalIntegrator._newSignalWeight = memoryTO.modeData.signalIntegrator.newSignalWeight;
            memory._mode = signalIntegrator;
        }
        memory._channelBitMask = memoryTO.channelBitMask;
        auto const& signalEntriesTO = getFromHeap<SignalEntryTO>(to.heap, memoryTO.signalEntriesDataIndex);
        copyMemoryEntriesToDescription(memory._signalEntries, signalEntriesTO, memoryTO.numSignalEntries);
        result._cellType = memory;
    } break;
    case CellType_Communicator: {
        CommunicatorDescription communicator;
        auto const& communicatorTO = cellTO.cellTypeData.communicator;
        if (communicatorTO.mode == CommunicatorMode_Sender) {
            SenderDescription sender;
            sender._range = communicatorTO.modeData.sender.range;
            sender._maxTimesSent = communicatorTO.modeData.sender.maxTimesSent;
            communicator._mode = sender;
        } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
            ReceiverDescription receiver;
            receiver._channelBitMask = communicatorTO.modeData.receiver.channelBitMask;
            receiver._restrictToColor = communicatorTO.modeData.receiver.restrictToColor != 255
                ? std::make_optional(static_cast<int>(communicatorTO.modeData.receiver.restrictToColor))
                : std::nullopt;
            receiver._restrictToLineage = communicatorTO.modeData.receiver.restrictToLineage;
            communicator._mode = receiver;
        }
        result._cellType = communicator;
    } break;
    }
    if (cellTO.neuralNetworkDataIndex != VALUE_NOT_SET_UINT64) {
        auto const& neuralNetworkTO = getFromHeap<NeuralNetworkTO>(to.heap, cellTO.neuralNetworkDataIndex);
        result._neuralNetwork = convert(*neuralNetworkTO);
    }

    SignalRestrictionDescription routingRestriction;
    routingRestriction._mode = cellTO.signalRestriction.mode;
    routingRestriction._baseAngle = cellTO.signalRestriction.baseAngle;
    routingRestriction._openingAngle = cellTO.signalRestriction.openingAngle;
    result._signalRestriction = routingRestriction;
    result._signalState = cellTO.signalState;
    if (cellTO.signalState == SignalState_Active) {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result._signal._channels[i] = cellTO.signal.channels[i];
        }
        result._signal._numTimesSent = cellTO.signal.numTimesSent;
    }
    result._activationTime = cellTO.activationTime;
    return result;
}


NodeDescription DescriptionConverterService::createNodeDescription(TO const& to, NodeTO const* nodeTO) const
{
    NodeDescription nodeDesc;
    nodeDesc._referenceAngle = nodeTO->referenceAngle;
    nodeDesc._color = nodeTO->color;
    nodeDesc._numAdditionalConnections = nodeTO->numAdditionalConnections;

    nodeDesc._neuralNetwork = convert(nodeTO->neuralNetwork);
    nodeDesc._signalRestriction._mode = nodeTO->signalRestriction.mode;
    nodeDesc._signalRestriction._baseAngle = nodeTO->signalRestriction.baseAngle;
    nodeDesc._signalRestriction._openingAngle = nodeTO->signalRestriction.openingAngle;

    switch (nodeTO->cellType) {
    case CellTypeGenome_Base: {
        BaseGenomeDescription baseDesc;
        nodeDesc._cellType = baseDesc;
    } break;
    case CellTypeGenome_Depot: {
        DepotGenomeDescription depotDesc;
        depotDesc._storageLimit = nodeTO->cellTypeData.depot.storageLimit;
        depotDesc._initialStoredUsableEnergy = nodeTO->cellTypeData.depot.initialStoredUsableEnergy;
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

        if (nodeTO->cellTypeData.sensor.mode == SensorMode_Telemetry) {
            TelemetryGenomeDescription telemetry;
            sensorDesc._mode = telemetry;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
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
        if (nodeTO->cellTypeData.attacker.mode == AttackerMode_FreeCell) {
            AttackFreeCellGenomeDescription attackFreeCell;
            attackFreeCell._restrictToColor = nodeTO->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor))
                : std::nullopt;
            attackerDesc._mode = attackFreeCell;
        } else if (nodeTO->cellTypeData.attacker.mode == AttackerMode_Creature) {
            AttackCreatureGenomeDescription attackCreature;
            attackCreature._minNumCells = nodeTO->cellTypeData.attacker.modeData.attackCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.attacker.modeData.attackCreature.minNumCells))
                : std::nullopt;
            attackCreature._maxNumCells = nodeTO->cellTypeData.attacker.modeData.attackCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.attacker.modeData.attackCreature.maxNumCells))
                : std::nullopt;
            attackCreature._restrictToColor = nodeTO->cellTypeData.attacker.modeData.attackCreature.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.attacker.modeData.attackCreature.restrictToColor))
                : std::nullopt;
            attackCreature._restrictToLineage = nodeTO->cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
            attackerDesc._mode = attackCreature;
        }
        nodeDesc._cellType = attackerDesc;
    } break;
    case CellTypeGenome_Injector: {
        InjectorGenomeDescription injectorDesc;
        injectorDesc._geneIndex = nodeTO->cellTypeData.injector.geneIndex;
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
        if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
            ReconnectStructureGenomeDescription reconnectStructure;
            reconnectorDesc._mode = reconnectStructure;
        } else if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
            ReconnectFreeCellGenomeDescription reconnectFreeCell;
            reconnectFreeCell._restrictToColor = nodeTO->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor))
                : std::nullopt;
            reconnectorDesc._mode = reconnectFreeCell;
        } else if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
            ReconnectCreatureGenomeDescription reconnectCreature;
            reconnectCreature._minNumCells = nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.minNumCells))
                : std::nullopt;
            reconnectCreature._maxNumCells = nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells))
                : std::nullopt;
            reconnectCreature._restrictToColor = nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor))
                : std::nullopt;
            reconnectCreature._restrictToLineage = nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
            reconnectorDesc._mode = reconnectCreature;
        }
        nodeDesc._cellType = reconnectorDesc;
    } break;
    case CellTypeGenome_Detonator: {
        DetonatorGenomeDescription detonatorDesc;
        detonatorDesc._countdown = nodeTO->cellTypeData.detonator.countdown;
        nodeDesc._cellType = detonatorDesc;
    } break;
    case CellTypeGenome_Digestor: {
        DigestorGenomeDescription digestorDesc;
        digestorDesc._rawEnergyConductivity = nodeTO->cellTypeData.digestor.rawEnergyConductivity;
        nodeDesc._cellType = digestorDesc;
    } break;
    case CellTypeGenome_Memory: {
        MemoryGenomeDescription memoryDesc;
        auto const& memoryTO = nodeTO->cellTypeData.memory;
        if (memoryTO.mode == MemoryMode_SignalDelay) {
            SignalDelayGenomeDescription signalDelay;
            signalDelay._delay = memoryTO.modeData.signalDelay.delay;
            memoryDesc._mode = signalDelay;
        } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
            SignalRecorderGenomeDescription signalRecorder;
            signalRecorder._readOnly = memoryTO.modeData.signalRecorder.readOnly;
            signalRecorder._numWrittenSignalEntries = memoryTO.modeData.signalRecorder.numWrittenSignalEntries;
            memoryDesc._mode = signalRecorder;
        } else if (memoryTO.mode == MemoryMode_SignalStorage) {
            SignalStorageGenomeDescription signalStorage;
            signalStorage._readOnly = memoryTO.modeData.signalStorage.readOnly;
            memoryDesc._mode = signalStorage;
        } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
            SignalIntegratorGenomeDescription signalIntegrator;
            signalIntegrator._newSignalWeight = memoryTO.modeData.signalIntegrator.newSignalWeight;
            memoryDesc._mode = signalIntegrator;
        }
        memoryDesc._channelBitMask = memoryTO.channelBitMask;
        auto const& signalEntriesTO = getFromHeap<SignalEntryGenomeTO>(to.heap, memoryTO.signalEntriesDataIndex);
        copyMemoryEntriesToDescription(memoryDesc._signalEntries, signalEntriesTO, memoryTO.numSignalEntries);
        nodeDesc._cellType = memoryDesc;
    } break;
    case CellTypeGenome_Communicator: {
        CommunicatorGenomeDescription communicatorDesc;
        auto const& communicatorTO = nodeTO->cellTypeData.communicator;
        if (communicatorTO.mode == CommunicatorMode_Sender) {
            SenderGenomeDescription sender;
            sender._range = communicatorTO.modeData.sender.range;
            sender._maxTimesSent = communicatorTO.modeData.sender.maxTimesSent;
            communicatorDesc._mode = sender;
        } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
            ReceiverGenomeDescription receiver;
            receiver._channelBitMask = communicatorTO.modeData.receiver.channelBitMask;
            receiver._restrictToColor = communicatorTO.modeData.receiver.restrictToColor != 255
                ? std::make_optional(static_cast<int>(communicatorTO.modeData.receiver.restrictToColor))
                : std::nullopt;
            receiver._restrictToLineage = communicatorTO.modeData.receiver.restrictToLineage;
            communicatorDesc._mode = receiver;
        }
        nodeDesc._cellType = communicatorDesc;
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
            geneDesc._nodes.emplace_back(createNodeDescription(to, nodeTO));
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
    std::vector<uint8_t>& heap,
    GenomeDescription const& genome,
    std::unordered_map<uint64_t, uint64_t>& genomeTOIndexById) const
{
    auto genomeTOIndex = genomeTOs.size();
    genomeTOs.resize(genomeTOIndex + 1);
    GenomeTO& genomeTO = genomeTOs.at(genomeTOIndex);

    genomeTOIndexById.insert_or_assign(genome._id, genomeTOIndex);

    auto geneArrayStartIndex = geneTOs.size();
    geneTOs.resize(geneArrayStartIndex + genome._genes.size());

    stringToChar64(genomeTO.name, genome._name);
    genomeTO.id = genome._id;
    genomeTO.frontAngle = genome._frontAngle;
    genomeTO.numGenes = toInt(genome._genes.size());
    genomeTO.geneArrayIndex = geneArrayStartIndex;
    genomeTO.genomeIndexOnGpu = VALUE_NOT_SET_UINT64;

    for (auto const& [geneIndex, geneDesc] : genome._genes | boost::adaptors::indexed(0)) {
        GeneTO& geneTO = geneTOs.at(geneArrayStartIndex + geneIndex);

        stringToChar64(geneTO.name, geneDesc._name);
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
            nodeTO.signalRestriction.mode = nodeDesc._signalRestriction._mode;
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
                depotTO.storageLimit = depotDesc._storageLimit;
                depotTO.initialStoredUsableEnergy = depotDesc._initialStoredUsableEnergy;
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
                sensorTO.minRange = static_cast<uint16_t>(sensorDesc._minRange);
                sensorTO.maxRange = static_cast<uint16_t>(sensorDesc._maxRange);
                sensorTO.mode = sensorDesc.getMode();

                if (sensorTO.mode == SensorMode_Telemetry) {
                    //auto const& telemetryDesc = std::get<TelemetryGenomeDescription>(sensorDesc._mode);
                    //auto& telemetryTO = sensorTO.modeData.telemetry;
                } else if (sensorTO.mode == SensorMode_DetectEnergy) {
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
                auto const& attackerDesc = std::get<AttackerGenomeDescription>(nodeDesc._cellType);
                auto& attackerTO = nodeTO.cellTypeData.attacker;
                attackerTO.mode = attackerDesc.getMode();
                if (attackerTO.mode == AttackerMode_FreeCell) {
                    auto const& attackFreeCellDesc = std::get<AttackFreeCellGenomeDescription>(attackerDesc._mode);
                    auto& attackFreeCellTO = attackerTO.modeData.attackFreeCell;
                    attackFreeCellTO.restrictToColor = static_cast<uint8_t>(attackFreeCellDesc._restrictToColor.value_or(255));
                } else if (attackerTO.mode == AttackerMode_Creature) {
                    auto const& attackCreatureDesc = std::get<AttackCreatureGenomeDescription>(attackerDesc._mode);
                    auto& attackCreatureTO = attackerTO.modeData.attackCreature;
                    attackCreatureTO.minNumCells = static_cast<uint32_t>(attackCreatureDesc._minNumCells.value_or(0));
                    attackCreatureTO.maxNumCells = static_cast<uint32_t>(attackCreatureDesc._maxNumCells.value_or(0));
                    attackCreatureTO.restrictToColor = static_cast<uint8_t>(attackCreatureDesc._restrictToColor.value_or(255));
                    attackCreatureTO.restrictToLineage = attackCreatureDesc._restrictToLineage;
                }
            } break;
            case CellTypeGenome_Injector: {
                auto const& injectorDesc = std::get<InjectorGenomeDescription>(nodeDesc._cellType);
                auto& injectorTO = nodeTO.cellTypeData.injector;
                injectorTO.geneIndex = injectorDesc._geneIndex;
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
                reconnectorTO.mode = reconnectorDesc.getMode();
                if (reconnectorTO.mode == ReconnectorMode_Structure) {
                    // No data to copy
                } else if (reconnectorTO.mode == ReconnectorMode_FreeCell) {
                    auto const& reconnectFreeCellDesc = std::get<ReconnectFreeCellGenomeDescription>(reconnectorDesc._mode);
                    auto& reconnectFreeCellTO = reconnectorTO.modeData.reconnectFreeCell;
                    reconnectFreeCellTO.restrictToColor = static_cast<uint8_t>(reconnectFreeCellDesc._restrictToColor.value_or(255));
                } else if (reconnectorTO.mode == ReconnectorMode_Creature) {
                    auto const& reconnectCreatureDesc = std::get<ReconnectCreatureGenomeDescription>(reconnectorDesc._mode);
                    auto& reconnectCreatureTO = reconnectorTO.modeData.reconnectCreature;
                    reconnectCreatureTO.minNumCells = static_cast<uint32_t>(reconnectCreatureDesc._minNumCells.value_or(0));
                    reconnectCreatureTO.maxNumCells = static_cast<uint32_t>(reconnectCreatureDesc._maxNumCells.value_or(0));
                    reconnectCreatureTO.restrictToColor = static_cast<uint8_t>(reconnectCreatureDesc._restrictToColor.value_or(255));
                    reconnectCreatureTO.restrictToLineage = reconnectCreatureDesc._restrictToLineage;
                }
            } break;
            case CellTypeGenome_Detonator: {
                auto const& detonatorDesc = std::get<DetonatorGenomeDescription>(nodeDesc._cellType);
                auto& detonatorTO = nodeTO.cellTypeData.detonator;
                detonatorTO.countdown = detonatorDesc._countdown;
            } break;
            case CellTypeGenome_Digestor: {
                auto const& digestorDesc = std::get<DigestorGenomeDescription>(nodeDesc._cellType);
                auto& digestorTO = nodeTO.cellTypeData.digestor;
                digestorTO.rawEnergyConductivity = digestorDesc._rawEnergyConductivity;
            } break;
            case CellTypeGenome_Memory: {
                auto const& memoryDesc = std::get<MemoryGenomeDescription>(nodeDesc._cellType);
                auto& memoryTO = nodeTO.cellTypeData.memory;
                memoryTO.mode = memoryDesc.getMode();
                if (memoryTO.mode == MemoryMode_SignalDelay) {
                    auto const& signalDelayDesc = std::get<SignalDelayGenomeDescription>(memoryDesc._mode);
                    memoryTO.modeData.signalDelay.delay = signalDelayDesc._delay;
                } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
                    auto const& signalRecorderDesc = std::get<SignalRecorderGenomeDescription>(memoryDesc._mode);
                    memoryTO.modeData.signalRecorder.readOnly = signalRecorderDesc._readOnly;
                    memoryTO.modeData.signalRecorder.numWrittenSignalEntries = signalRecorderDesc._numWrittenSignalEntries;
                } else if (memoryTO.mode == MemoryMode_SignalStorage) {
                    auto const& signalStorageDesc = std::get<SignalStorageGenomeDescription>(memoryDesc._mode);
                    memoryTO.modeData.signalStorage.readOnly = signalStorageDesc._readOnly;
                } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
                    auto const& signalIntegratorDesc = std::get<SignalIntegratorGenomeDescription>(memoryDesc._mode);
                    memoryTO.modeData.signalIntegrator.newSignalWeight = signalIntegratorDesc._newSignalWeight;
                }
                memoryTO.channelBitMask = memoryDesc._channelBitMask;
                memoryTO.numSignalEntries = toInt(memoryDesc._signalEntries.size());
                memoryTO.signalEntriesDataIndex = heap.size();
                heap.resize(heap.size() + sizeof(SignalEntryGenomeTO) * memoryTO.numSignalEntries);
                auto signalEntriesTO = reinterpret_cast<SignalEntryGenomeTO*>(heap.data() + memoryTO.signalEntriesDataIndex);
                copyMemoryEntriesToTO(signalEntriesTO, memoryDesc._signalEntries);
            } break;
            case CellTypeGenome_Communicator: {
                auto const& communicatorDesc = std::get<CommunicatorGenomeDescription>(nodeDesc._cellType);
                auto& communicatorTO = nodeTO.cellTypeData.communicator;
                communicatorTO.mode = communicatorDesc.getMode();
                if (communicatorTO.mode == CommunicatorMode_Sender) {
                    auto const& senderDesc = std::get<SenderGenomeDescription>(communicatorDesc._mode);
                    communicatorTO.modeData.sender.range = senderDesc._range;
                    communicatorTO.modeData.sender.maxTimesSent = senderDesc._maxTimesSent;
                } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
                    auto const& receiverDesc = std::get<ReceiverGenomeDescription>(communicatorDesc._mode);
                    communicatorTO.modeData.receiver.channelBitMask = receiverDesc._channelBitMask;
                    communicatorTO.modeData.receiver.restrictToColor = static_cast<uint8_t>(receiverDesc._restrictToColor.value_or(255));
                    communicatorTO.modeData.receiver.restrictToLineage = receiverDesc._restrictToLineage;
                }
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
    cellTO.usableEnergy = cellDesc._usableEnergy;
    checkAndCorrectInvalidEnergy(cellTO.usableEnergy);
    cellTO.rawEnergy = cellDesc._rawEnergy;
    cellTO.stiffness = cellDesc._stiffness;
    cellTO.cellState = cellDesc._cellState;
    cellTO.cellType = cellDesc.getCellType();
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
        auto const& depotDesc = std::get<DepotDescription>(cellDesc._cellType);
        DepotTO& depotTO = cellTO.cellTypeData.depot;
        depotTO.storageLimit = depotDesc._storageLimit;
        depotTO.storedUsableEnergy = depotDesc._storedUsableEnergy;
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
        sensorTO.minRange = static_cast<uint16_t>(sensorDesc._minRange);
        sensorTO.maxRange = static_cast<uint16_t>(sensorDesc._maxRange);
        sensorTO.mode = sensorDesc.getMode();

        if (sensorTO.mode == SensorMode_Telemetry) {
            //auto const& telemetryDesc = std::get<TelemetryDescription>(sensorDesc._mode);
            //TelemetryTO& telemetryTO = sensorTO.modeData.telemetry;
        } else if (sensorTO.mode == SensorMode_DetectEnergy) {
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
        }
        sensorTO.lastMatchAvailable = sensorDesc._lastMatch.has_value();
        if (sensorDesc._lastMatch.has_value()) {
            sensorTO.lastMatch.creatureId = sensorDesc._lastMatch->_creatureId;
            sensorTO.lastMatch.pos = {sensorDesc._lastMatch->_pos.x, sensorDesc._lastMatch->_pos.y};
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
        auto const& attackerDesc = std::get<AttackerDescription>(cellDesc._cellType);
        AttackerTO& attackerTO = cellTO.cellTypeData.attacker;
        attackerTO.mode = attackerDesc.getMode();
        if (attackerTO.mode == AttackerMode_FreeCell) {
            auto const& attackFreeCellDesc = std::get<AttackFreeCellDescription>(attackerDesc._mode);
            attackerTO.modeData.attackFreeCell.restrictToColor = static_cast<uint8_t>(attackFreeCellDesc._restrictToColor.value_or(255));
        } else if (attackerTO.mode == AttackerMode_Creature) {
            auto const& attackCreatureDesc = std::get<AttackCreatureDescription>(attackerDesc._mode);
            attackerTO.modeData.attackCreature.minNumCells = static_cast<uint32_t>(attackCreatureDesc._minNumCells.value_or(0));
            attackerTO.modeData.attackCreature.maxNumCells = static_cast<uint32_t>(attackCreatureDesc._maxNumCells.value_or(0));
            attackerTO.modeData.attackCreature.restrictToColor = static_cast<uint8_t>(attackCreatureDesc._restrictToColor.value_or(255));
            attackerTO.modeData.attackCreature.restrictToLineage = attackCreatureDesc._restrictToLineage;
        }
    } break;
    case CellType_Injector: {
        auto const& injectorDesc = std::get<InjectorDescription>(cellDesc._cellType);
        InjectorTO& injectorTO = cellTO.cellTypeData.injector;
        injectorTO.geneIndex = static_cast<uint16_t>(injectorDesc._geneIndex);
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
        reconnectorTO.mode = reconnectorDesc.getMode();
        if (reconnectorTO.mode == ReconnectorMode_Structure) {
            // No data to copy
        } else if (reconnectorTO.mode == ReconnectorMode_FreeCell) {
            auto const& reconnectFreeCellDesc = std::get<ReconnectFreeCellDescription>(reconnectorDesc._mode);
            ReconnectFreeCellTO& reconnectFreeCellTO = reconnectorTO.modeData.reconnectFreeCell;
            reconnectFreeCellTO.restrictToColor = static_cast<uint8_t>(reconnectFreeCellDesc._restrictToColor.value_or(255));
        } else if (reconnectorTO.mode == ReconnectorMode_Creature) {
            auto const& reconnectCreatureDesc = std::get<ReconnectCreatureDescription>(reconnectorDesc._mode);
            ReconnectCreatureTO& reconnectCreatureTO = reconnectorTO.modeData.reconnectCreature;
            reconnectCreatureTO.minNumCells = static_cast<uint32_t>(reconnectCreatureDesc._minNumCells.value_or(0));
            reconnectCreatureTO.maxNumCells = static_cast<uint32_t>(reconnectCreatureDesc._maxNumCells.value_or(0));
            reconnectCreatureTO.restrictToColor = static_cast<uint8_t>(reconnectCreatureDesc._restrictToColor.value_or(255));
            reconnectCreatureTO.restrictToLineage = reconnectCreatureDesc._restrictToLineage;
        }
    } break;
    case CellType_Detonator: {
        auto const& detonatorDesc = std::get<DetonatorDescription>(cellDesc._cellType);
        DetonatorTO& detonatorTO = cellTO.cellTypeData.detonator;
        detonatorTO.state = detonatorDesc._state;
        detonatorTO.countdown = detonatorDesc._countdown;
    } break;
    case CellType_Digestor: {
        auto const& digestorDesc = std::get<DigestorDescription>(cellDesc._cellType);
        DigestorTO& digestorTO = cellTO.cellTypeData.digestor;
        digestorTO.rawEnergyConductivity = digestorDesc._rawEnergyConductivity;
    } break;
    case CellType_Memory: {
        auto const& memoryDesc = std::get<MemoryDescription>(cellDesc._cellType);
        auto& memoryTO = cellTO.cellTypeData.memory;
        memoryTO.mode = memoryDesc.getMode();
        if (memoryTO.mode == MemoryMode_SignalDelay) {
            auto const& signalDelayDesc = std::get<SignalDelayDescription>(memoryDesc._mode);
            memoryTO.modeData.signalDelay.delay = signalDelayDesc._delay;
            memoryTO.modeData.signalDelay.numSignalEntriesInitialized = signalDelayDesc._numSignalEntriesInitialized;
            memoryTO.modeData.signalDelay.ringBufferIndex = signalDelayDesc._ringBufferIndex;
        } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
            auto const& signalRecorderDesc = std::get<SignalRecorderDescription>(memoryDesc._mode);
            memoryTO.modeData.signalRecorder.readOnly = signalRecorderDesc._readOnly;
            memoryTO.modeData.signalRecorder.state = signalRecorderDesc._state;
            memoryTO.modeData.signalRecorder.numWrittenSignalEntries = signalRecorderDesc._numWrittenSignalEntries;
            memoryTO.modeData.signalRecorder.numReadSignalEntries = signalRecorderDesc._numReadSignalEntries;
        } else if (memoryTO.mode == MemoryMode_SignalStorage) {
            auto const& signalStorageDesc = std::get<SignalStorageDescription>(memoryDesc._mode);
            memoryTO.modeData.signalStorage.readOnly = signalStorageDesc._readOnly;
        } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
            auto const& signalIntegratorDesc = std::get<SignalIntegratorDescription>(memoryDesc._mode);
            memoryTO.modeData.signalIntegrator.newSignalWeight = signalIntegratorDesc._newSignalWeight;
        }
        memoryTO.channelBitMask = memoryDesc._channelBitMask;
        memoryTO.numSignalEntries = toInt(memoryDesc._signalEntries.size());
        memoryTO.signalEntriesDataIndex = heap.size();
        heap.resize(heap.size() + sizeof(SignalEntryTO) * memoryTO.numSignalEntries);
        auto signalEntriesTO = reinterpret_cast<SignalEntryTO*>(heap.data() + memoryTO.signalEntriesDataIndex);
        copyMemoryEntriesToTO(signalEntriesTO, memoryDesc._signalEntries);
    } break;
    case CellType_Communicator: {
        auto const& communicatorDesc = std::get<CommunicatorDescription>(cellDesc._cellType);
        CommunicatorTO& communicatorTO = cellTO.cellTypeData.communicator;
        communicatorTO.mode = communicatorDesc.getMode();
        if (communicatorTO.mode == CommunicatorMode_Sender) {
            auto const& senderDesc = std::get<SenderDescription>(communicatorDesc._mode);
            communicatorTO.modeData.sender.range = senderDesc._range;
            communicatorTO.modeData.sender.maxTimesSent = senderDesc._maxTimesSent;
        } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
            auto const& receiverDesc = std::get<ReceiverDescription>(communicatorDesc._mode);
            communicatorTO.modeData.receiver.channelBitMask = receiverDesc._channelBitMask;
            communicatorTO.modeData.receiver.restrictToColor = static_cast<uint8_t>(receiverDesc._restrictToColor.value_or(255));
            communicatorTO.modeData.receiver.restrictToLineage = receiverDesc._restrictToLineage;
        }
    } break;
    }
    cellTO.signalRestriction.mode = cellDesc._signalRestriction._mode;
    cellTO.signalRestriction.baseAngle = cellDesc._signalRestriction._baseAngle;
    cellTO.signalRestriction.openingAngle = cellDesc._signalRestriction._openingAngle;
    cellTO.signalState = cellDesc._signalState;
    if (cellDesc._signalState == SignalState_Active) {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cellDesc._signal._channels[i];
        }
        cellTO.signal.numTimesSent = cellDesc._signal._numTimesSent;
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
