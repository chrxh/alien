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

    // Helper function to copy memory entries from Desc to TO
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

    NeuralNetworkGenomeDesc convert(NeuralNetworkGenomeTO const& neuralNetworkGenomeTO)
    {
        NeuralNetworkGenomeDesc result;
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

    NeuralNetworkDesc convert(NeuralNetworkTO const& neuralNetworkTO)
    {
        NeuralNetworkDesc result;
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

    NeuralNetworkGenomeTO convert(NeuralNetworkGenomeDesc const& neuralNetworkDesc)
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

    NeuralNetworkTO convert(NeuralNetworkDesc const& neuralNetworkDesc)
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

Desc DescriptionConverterService::convertTOtoDescription(TOs const& to) const
{
    Desc result;

    // Genomes
    for (int i = 0; i < *to.numGenomes; ++i) {
        auto genome = createGenomeDesc(to, i);
        result._genomes.emplace_back(genome);
    }

    // Creatures
    std::unordered_map<uint64_t, uint64_t> creatureIdByTOIndex;
    for (int i = 0; i < *to.numCreatures; ++i) {
        auto creature = createCreatureDesc(to, i);
        auto genomeTOIndex = to.creatures[i].genomeArrayIndex;
        auto genomeId = to.genomes[genomeTOIndex].id;
        creature._genomeId = genomeId;

        creatureIdByTOIndex.emplace(i, creature._id);
        result._creatures.emplace_back(creature);
    }

    // Cells
    for (int i = 0; i < *to.numObjects; ++i) {
        auto object = createObjectDesc(to, i);

        // Only access cell data for Cell objects
        if (to.objects[i].type == ObjectType_Cell) {
            auto creatureTOIndex = to.objects[i].typeData.cell.creatureIndex;
            object.getCellRef()._creatureId = creatureIdByTOIndex.at(creatureTOIndex);
        }
        result._objects.emplace_back(object);
    }

    // Particles
    for (int i = 0; i < *to.numEnergyParticles; ++i) {
        result._energies.emplace_back(createEnergyDesc(to, i));
    }

    return result;
}

TOs DescriptionConverterService::convertDescriptionToTO(Desc const& description) const
{
    std::vector<GenomeTO> genomeTOs;
    std::vector<CreatureTO> creatureTOs;
    std::vector<GeneTO> geneTOs;
    std::vector<NodeTO> nodeTOs;
    std::vector<ObjectTO> objectTOs;
    std::vector<EnergyTO> particleTOs;
    std::vector<uint8_t> heap;

    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    for (auto const& genome : description._genomes) {
        convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, heap, genome, genomeTOIndexById);
    }

    std::unordered_map<uint64_t, uint64_t> creatureTOIndexById;
    for (auto const& creature : description._creatures) {
        convertCreatureToTO(creatureTOs, creature, genomeTOIndexById, creatureTOIndexById);
    }

    std::unordered_map<uint64_t, uint64_t> objectIndexTOById;
    for (auto const& object : description._objects) {
        convertObjectToTO(objectTOs, heap, objectIndexTOById, object, creatureTOIndexById);
    }
    for (auto const& object : description._objects) {
        setConnections(objectTOs, object, objectIndexTOById);
    }
    for (auto const& energyParticle : description._energies) {
        addParticle(particleTOs, energyParticle);
    }

    return provideDataTO(creatureTOs, genomeTOs, geneTOs, nodeTOs, objectTOs, particleTOs, heap);
}

TOs DescriptionConverterService::convertDescriptionToTO(ObjectDesc const& object) const
{
    std::vector<ObjectTO> objectTOs;
    std::vector<uint8_t> heap;

    std::unordered_map<uint64_t, uint64_t> objectIndexTOById;
    std::unordered_map<uint64_t, uint64_t> creatureTOIndexById;
    convertObjectToTO(objectTOs, heap, objectIndexTOById, object, creatureTOIndexById);

    return provideDataTO({}, {}, {}, {}, objectTOs, {}, heap);
}

TOs DescriptionConverterService::convertDescriptionToTO(EnergyDesc const& particle) const
{
    std::vector<EnergyTO> particleTOs;
    std::vector<uint8_t> heap;
    addParticle(particleTOs, particle);

    return provideDataTO({}, {}, {}, {}, {}, particleTOs, heap);
}

TOs DescriptionConverterService::convertDescriptionToTO(uint64_t creatureId, GenomeDesc const& genome) const
{
    std::vector<GenomeTO> genomeTOs;
    std::vector<CreatureTO> creatureTOs;
    std::vector<GeneTO> geneTOs;
    std::vector<NodeTO> nodeTOs;
    std::vector<uint8_t> heap;

    auto wrapper = CreatureDesc().id(creatureId).genomeId(genome._id);

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

ObjectDesc DescriptionConverterService::createObjectDesc(TOs const& to, int objectIndex) const
{
    ObjectDesc result(false);

    auto const& objectTO = to.objects[objectIndex];
    result._id = objectTO.id;
    NumberGenerator::get().adaptMaxIds({.entityId = objectTO.id});
    result._pos = RealVector2D{objectTO.pos.x, objectTO.pos.y};
    result._vel = RealVector2D{objectTO.vel.x, objectTO.vel.y};
    result._stiffness = objectTO.stiffness;
    std::vector<ConnectionDesc> connections;
    for (int i = 0; i < objectTO.numConnections; ++i) {
        auto const& connectionTO = objectTO.connections[i];
        ConnectionDesc connection;
        if (connectionTO.objectIndex != VALUE_NOT_SET_UINT64) {
            connection._objectId = to.objects[connectionTO.objectIndex].id;
        } else {
            connections.clear();
            break;
        }
        connection._distance = connectionTO.distance;
        connection._angleFromPrevious = connectionTO.angleFromPrevious;
        connections.emplace_back(connection);
    }
    result._connections = connections;
    result._fixed = objectTO.fixed;
    result._sticky = objectTO.sticky;
    result._color = objectTO.color;

    // Handle object type: Structure, FreeCell, or Cell
    if (objectTO.type == ObjectType_Structure) {
        StructureDesc structureDesc;
        structureDesc._energy = objectTO.typeData.structure.energy;
        result._type = structureDesc;

    } else if (objectTO.type == ObjectType_FreeCell) {
        FreeCellDesc freeCellDesc;
        freeCellDesc._energy = objectTO.typeData.freeCell.energy;
        freeCellDesc._age = objectTO.typeData.freeCell.age;
        result._type = freeCellDesc;

    } else if (objectTO.type == ObjectType_Cell) {

        // ObjectType_Cell - access typeData.cell
        CellDesc cellDesc;
        cellDesc._usableEnergy = objectTO.typeData.cell.usableEnergy;
        cellDesc._rawEnergy = objectTO.typeData.cell.rawEnergy;
        cellDesc._cellState = objectTO.typeData.cell.cellState;
        cellDesc._age = objectTO.typeData.cell.age;
        cellDesc._frontAngle = objectTO.typeData.cell.frontAngle != VALUE_NOT_SET_FLOAT ? std::make_optional(objectTO.typeData.cell.frontAngle) : std::nullopt;
        cellDesc._nodeIndex = objectTO.typeData.cell.nodeIndex;
        cellDesc._parentNodeIndex = objectTO.typeData.cell.parentNodeIndex;
        cellDesc._geneIndex = objectTO.typeData.cell.geneIndex;
        cellDesc._frontAngleId = objectTO.typeData.cell.frontAngleId;
        cellDesc._headCell = objectTO.typeData.cell.headCell;
        cellDesc._event = objectTO.typeData.cell.event;
        cellDesc._eventCounter = objectTO.typeData.cell.eventCounter;
        cellDesc._eventPos = {objectTO.typeData.cell.eventPos.x, objectTO.typeData.cell.eventPos.y};

        switch (objectTO.typeData.cell.cellType) {
        case CellType_Base: {
            BaseDesc base;
            cellDesc._cellType = base;
        } break;
        case CellType_Depot: {
            DepotDesc transmitter;
            transmitter._storageLimit = objectTO.typeData.cell.cellTypeData.depot.storageLimit;
            transmitter._storedUsableEnergy = objectTO.typeData.cell.cellTypeData.depot.storedUsableEnergy;
            cellDesc._cellType = transmitter;
        } break;
        case CellType_Constructor: {
            ConstructorDesc constructor;
            constructor._autoTriggerInterval = objectTO.typeData.cell.cellTypeData.constructor.autoTriggerInterval > 0
                ? std::make_optional(objectTO.typeData.cell.cellTypeData.constructor.autoTriggerInterval)
                : std::nullopt;
            constructor._constructionActivationTime = objectTO.typeData.cell.cellTypeData.constructor.constructionActivationTime;
            constructor._constructionAngle = objectTO.typeData.cell.cellTypeData.constructor.constructionAngle;
            constructor._provideEnergy = objectTO.typeData.cell.cellTypeData.constructor.provideEnergy;
            constructor._geneIndex = objectTO.typeData.cell.cellTypeData.constructor.geneIndex;
            constructor._lastConstructedCellId = objectTO.typeData.cell.cellTypeData.constructor.lastConstructedCellId != VALUE_NOT_SET_UINT64
                ? std::make_optional(objectTO.typeData.cell.cellTypeData.constructor.lastConstructedCellId)
                : std::nullopt;
            constructor._currentNodeIndex = objectTO.typeData.cell.cellTypeData.constructor.currentNodeIndex;
            constructor._currentConcatenation = objectTO.typeData.cell.cellTypeData.constructor.currentConcatenation;
            constructor._currentBranch = objectTO.typeData.cell.cellTypeData.constructor.currentBranch;
            cellDesc._cellType = constructor;
        } break;
        case CellType_Sensor: {
            SensorDesc sensor;
            sensor._autoTriggerInterval = objectTO.typeData.cell.cellTypeData.sensor.autoTriggerInterval > 0
                ? std::make_optional(objectTO.typeData.cell.cellTypeData.sensor.autoTriggerInterval)
                : std::nullopt;
            sensor._minRange = objectTO.typeData.cell.cellTypeData.sensor.minRange;
            sensor._maxRange = objectTO.typeData.cell.cellTypeData.sensor.maxRange;

            if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_Telemetry) {
                TelemetryDesc telemetry;
                sensor._mode = telemetry;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                DetectEnergyDesc detectEnergy;
                detectEnergy._minDensity = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectEnergy.minDensity;
                sensor._mode = detectEnergy;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectStructure) {
                DetectStructureDesc detectStructure;
                sensor._mode = detectStructure;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                DetectFreeCellDesc detectFreeCell;
                detectFreeCell._minDensity = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                detectFreeCell._restrictToColor = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor != 255
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.restrictToColor))
                    : std::nullopt;
                sensor._mode = detectFreeCell;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                DetectCreatureDesc detectCreature;
                detectCreature._minNumCells = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.minNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.minNumCells))
                    : std::nullopt;
                detectCreature._maxNumCells = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells))
                    : std::nullopt;
                detectCreature._restrictToColor = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToColor != 255
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToColor))
                    : std::nullopt;
                detectCreature._restrictToLineage = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
                sensor._mode = detectCreature;
            }
            if (objectTO.typeData.cell.cellTypeData.sensor.lastMatchAvailable) {
                SensorLastMatchDesc lastMatchDesc;
                lastMatchDesc._creatureId = objectTO.typeData.cell.cellTypeData.sensor.lastMatch.creatureId;
                lastMatchDesc._pos =
                    RealVector2D{objectTO.typeData.cell.cellTypeData.sensor.lastMatch.pos.x, objectTO.typeData.cell.cellTypeData.sensor.lastMatch.pos.y};
                sensor._lastMatch = lastMatchDesc;
            }

            cellDesc._cellType = sensor;
        } break;
        case CellType_Generator: {
            GeneratorDesc generator;
            generator._autoTriggerInterval = objectTO.typeData.cell.cellTypeData.generator.autoTriggerInterval;
            generator._pulseType = objectTO.typeData.cell.cellTypeData.generator.pulseType;
            generator._alternationInterval = objectTO.typeData.cell.cellTypeData.generator.alternationInterval;
            generator._numPulses = objectTO.typeData.cell.cellTypeData.generator.numPulses;
            cellDesc._cellType = generator;
        } break;
        case CellType_Attacker: {
            AttackerDesc attacker;
            if (objectTO.typeData.cell.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                AttackFreeCellDesc attackFreeCell;
                attackFreeCell._restrictToColor = objectTO.typeData.cell.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor != 255
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor))
                    : std::nullopt;
                attacker._mode = attackFreeCell;
            } else if (objectTO.typeData.cell.cellTypeData.attacker.mode == AttackerMode_Creature) {
                AttackCreatureDesc attackCreature;
                attackCreature._minNumCells = objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.minNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.minNumCells))
                    : std::nullopt;
                attackCreature._maxNumCells = objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.maxNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.maxNumCells))
                    : std::nullopt;
                attackCreature._restrictToColor = objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToColor != 255
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToColor))
                    : std::nullopt;
                attackCreature._restrictToLineage = objectTO.typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToLineage;
                attacker._mode = attackCreature;
            }
            cellDesc._cellType = attacker;
        } break;
        case CellType_Injector: {
            InjectorDesc injector;
            injector._geneIndex = objectTO.typeData.cell.cellTypeData.injector.geneIndex;
            cellDesc._cellType = injector;
        } break;
        case CellType_Muscle: {
            MuscleDesc muscle;
            if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
                AutoBendingDesc bending;
                bending._maxAngleDeviation = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                bending._forwardBackwardRatio = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
                bending._initialAngle = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.initialAngle != VALUE_NOT_SET_FLOAT
                    ? std::make_optional(objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.initialAngle)
                    : std::nullopt;
                bending._forward = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.forward;
                bending._activation = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.activation;
                bending._activationCountdown = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.activationCountdown;
                bending._impulseAlreadyApplied = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
                muscle._mode = bending;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                ManualBendingDesc bending;
                bending._maxAngleDeviation = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                bending._forwardBackwardRatio = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                bending._initialAngle = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle != VALUE_NOT_SET_FLOAT
                    ? std::make_optional(objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle)
                    : std::nullopt;
                bending._lastAngleDelta = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
                bending._impulseAlreadyApplied = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
                muscle._mode = bending;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
                AngleBendingDesc bending;
                bending._maxAngleDeviation = objectTO.typeData.cell.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                bending._attractionRepulsionRatio = objectTO.typeData.cell.cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
                bending._initialAngle = objectTO.typeData.cell.cellTypeData.muscle.modeData.angleBending.initialAngle != VALUE_NOT_SET_FLOAT
                    ? std::make_optional(objectTO.typeData.cell.cellTypeData.muscle.modeData.angleBending.initialAngle)
                    : std::nullopt;
                muscle._mode = bending;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
                AutoCrawlingDesc crawling;
                crawling._maxDistanceDeviation = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                crawling._forwardBackwardRatio = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
                crawling._initialDistance = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.initialDistance != VALUE_NOT_SET_FLOAT
                    ? std::make_optional(objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.initialDistance)
                    : std::nullopt;
                crawling._lastActualDistance = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
                crawling._forward = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.forward;
                crawling._activation = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.activation;
                crawling._activationCountdown = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
                crawling._impulseAlreadyApplied = objectTO.typeData.cell.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
                muscle._mode = crawling;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
                ManualCrawlingDesc crawling;
                crawling._maxDistanceDeviation = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                crawling._forwardBackwardRatio = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
                crawling._initialDistance = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.initialDistance != VALUE_NOT_SET_FLOAT
                    ? std::make_optional(objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.initialDistance)
                    : std::nullopt;
                crawling._lastActualDistance = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.lastActualDistance;
                crawling._lastDistanceDelta = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.lastDistanceDelta;
                crawling._impulseAlreadyApplied = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualCrawling.impulseAlreadyApplied;
                muscle._mode = crawling;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_DirectMovement) {
                DirectMovementDesc movement;
                muscle._mode = movement;
            }

            muscle._lastMovementX = objectTO.typeData.cell.cellTypeData.muscle.lastMovementX;
            muscle._lastMovementY = objectTO.typeData.cell.cellTypeData.muscle.lastMovementY;
            cellDesc._cellType = muscle;
        } break;
        case CellType_Defender: {
            DefenderDesc defender;
            defender._mode = objectTO.typeData.cell.cellTypeData.defender.mode;
            cellDesc._cellType = defender;
        } break;
        case CellType_Reconnector: {
            ReconnectorDesc reconnector;
            if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
                ReconnectStructureDesc reconnectStructure;
                reconnector._mode = reconnectStructure;
            } else if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                ReconnectFreeCellDesc reconnectFreeCell;
                reconnectFreeCell._restrictToColor = objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor != 255
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor))
                    : std::nullopt;
                reconnector._mode = reconnectFreeCell;
            } else if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                ReconnectCreatureDesc reconnectCreature;
                reconnectCreature._minNumCells = objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells))
                    : std::nullopt;
                reconnectCreature._maxNumCells = objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells))
                    : std::nullopt;
                reconnectCreature._restrictToColor = objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor != 255
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColor))
                    : std::nullopt;
                reconnectCreature._restrictToLineage = objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToLineage;
                reconnector._mode = reconnectCreature;
            }
            cellDesc._cellType = reconnector;
        } break;
        case CellType_Detonator: {
            DetonatorDesc detonator;
            detonator._state = objectTO.typeData.cell.cellTypeData.detonator.state;
            detonator._countdown = objectTO.typeData.cell.cellTypeData.detonator.countdown;
            cellDesc._cellType = detonator;
        } break;
        case CellType_Digestor: {
            DigestorDesc digestor;
            digestor._rawEnergyConductivity = objectTO.typeData.cell.cellTypeData.digestor.rawEnergyConductivity;
            cellDesc._cellType = digestor;
        } break;
        case CellType_Memory: {
            MemoryDesc memory;
            auto const& memoryTO = objectTO.typeData.cell.cellTypeData.memory;
            if (memoryTO.mode == MemoryMode_SignalDelay) {
                SignalDelayDesc signalDelay;
                signalDelay._delay = memoryTO.modeData.signalDelay.delay;
                signalDelay._numSignalEntriesInitialized = memoryTO.modeData.signalDelay.numSignalEntriesInitialized;
                signalDelay._ringBufferIndex = memoryTO.modeData.signalDelay.ringBufferIndex;
                memory._mode = signalDelay;
            } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
                SignalRecorderDesc signalRecorder;
                signalRecorder._readOnly = memoryTO.modeData.signalRecorder.readOnly;
                signalRecorder._state = memoryTO.modeData.signalRecorder.state;
                signalRecorder._numWrittenSignalEntries = memoryTO.modeData.signalRecorder.numWrittenSignalEntries;
                signalRecorder._numReadSignalEntries = memoryTO.modeData.signalRecorder.numReadSignalEntries;
                memory._mode = signalRecorder;
            } else if (memoryTO.mode == MemoryMode_SignalStorage) {
                SignalStorageDesc signalStorage;
                signalStorage._readOnly = memoryTO.modeData.signalStorage.readOnly;
                memory._mode = signalStorage;
            } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
                SignalIntegratorDesc signalIntegrator;
                signalIntegrator._newSignalWeight = memoryTO.modeData.signalIntegrator.newSignalWeight;
                memory._mode = signalIntegrator;
            }
            memory._channelBitMask = memoryTO.channelBitMask;
            auto const& signalEntriesTO = getFromHeap<SignalEntryTO>(to.heap, memoryTO.signalEntriesDataIndex);
            copyMemoryEntriesToDescription(memory._signalEntries, signalEntriesTO, memoryTO.numSignalEntries);
            cellDesc._cellType = memory;
        } break;
        case CellType_Communicator: {
            CommunicatorDesc communicator;
            auto const& communicatorTO = objectTO.typeData.cell.cellTypeData.communicator;
            if (communicatorTO.mode == CommunicatorMode_Sender) {
                SenderDesc sender;
                sender._range = communicatorTO.modeData.sender.range;
                sender._maxTimesSent = communicatorTO.modeData.sender.maxTimesSent;
                communicator._mode = sender;
            } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
                ReceiverDesc receiver;
                receiver._restrictToColor = communicatorTO.modeData.receiver.restrictToColor != 255
                    ? std::make_optional(static_cast<int>(communicatorTO.modeData.receiver.restrictToColor))
                    : std::nullopt;
                receiver._restrictToLineage = communicatorTO.modeData.receiver.restrictToLineage;
                communicator._mode = receiver;
            }
            cellDesc._cellType = communicator;
        } break;
        }
        auto const& neuralNetworkTO = getFromHeap<NeuralNetworkTO>(to.heap, objectTO.typeData.cell.neuralNetworkDataIndex);
        cellDesc._neuralNetwork = convert(*neuralNetworkTO);

        SignalRestrictionDesc routingRestriction;
        routingRestriction._mode = objectTO.typeData.cell.signalRestriction.mode;
        routingRestriction._baseAngle = objectTO.typeData.cell.signalRestriction.baseAngle;
        routingRestriction._openingAngle = objectTO.typeData.cell.signalRestriction.openingAngle;
        cellDesc._signalRestriction = routingRestriction;
        cellDesc._signalState = objectTO.typeData.cell.signalState;
        if (objectTO.typeData.cell.signalState == SignalState_Active) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cellDesc._signal._channels[i] = objectTO.typeData.cell.signal.channels[i];
            }
            cellDesc._signal._numTimesSent = objectTO.typeData.cell.signal.numTimesSent;
        }
        cellDesc._activationTime = objectTO.typeData.cell.activationTime;
        result._type = cellDesc;

    } else {
        CHECK(false);
    }
    return result;
}


NodeDesc DescriptionConverterService::createNodeDesc(TOs const& to, NodeTO const* nodeTO) const
{
    NodeDesc nodeDesc;
    nodeDesc._referenceAngle = nodeTO->referenceAngle;
    nodeDesc._color = nodeTO->color;
    nodeDesc._numAdditionalConnections = nodeTO->numAdditionalConnections;

    nodeDesc._neuralNetwork = convert(nodeTO->neuralNetwork);
    nodeDesc._signalRestriction._mode = nodeTO->signalRestriction.mode;
    nodeDesc._signalRestriction._baseAngle = nodeTO->signalRestriction.baseAngle;
    nodeDesc._signalRestriction._openingAngle = nodeTO->signalRestriction.openingAngle;

    switch (nodeTO->cellType) {
    case CellType_Base: {
        BaseGenomeDesc baseDesc;
        nodeDesc._cellType = baseDesc;
    } break;
    case CellType_Depot: {
        DepotGenomeDesc depotDesc;
        depotDesc._storageLimit = nodeTO->cellTypeData.depot.storageLimit;
        depotDesc._initialStoredUsableEnergy = nodeTO->cellTypeData.depot.initialStoredUsableEnergy;
        nodeDesc._cellType = depotDesc;
    } break;
    case CellType_Constructor: {
        ConstructorGenomeDesc constructorDesc;
        constructorDesc._autoTriggerInterval =
            nodeTO->cellTypeData.constructor.autoTriggerInterval > 0 ? std::make_optional(nodeTO->cellTypeData.constructor.autoTriggerInterval) : std::nullopt;
        constructorDesc._geneIndex = nodeTO->cellTypeData.constructor.geneIndex;
        constructorDesc._constructionActivationTime = nodeTO->cellTypeData.constructor.constructionActivationTime;
        constructorDesc._constructionAngle = nodeTO->cellTypeData.constructor.constructionAngle;
        constructorDesc._provideEnergy = nodeTO->cellTypeData.constructor.provideEnergy;
        nodeDesc._cellType = constructorDesc;
    } break;
    case CellType_Sensor: {
        SensorGenomeDesc sensorDesc;
        sensorDesc._autoTriggerInterval =
            nodeTO->cellTypeData.sensor.autoTriggerInterval > 0 ? std::make_optional(nodeTO->cellTypeData.sensor.autoTriggerInterval) : std::nullopt;
        sensorDesc._minRange = nodeTO->cellTypeData.sensor.minRange;
        sensorDesc._maxRange = nodeTO->cellTypeData.sensor.maxRange;

        if (nodeTO->cellTypeData.sensor.mode == SensorMode_Telemetry) {
            TelemetryGenomeDesc telemetry;
            sensorDesc._mode = telemetry;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
            DetectEnergyGenomeDesc detectEnergy;
            detectEnergy._minDensity = nodeTO->cellTypeData.sensor.modeData.detectEnergy.minDensity;
            sensorDesc._mode = detectEnergy;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectStructure) {
            DetectStructureGenomeDesc detectStructure;
            sensorDesc._mode = detectStructure;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
            DetectFreeCellGenomeDesc detectFreeCell;
            detectFreeCell._minDensity = nodeTO->cellTypeData.sensor.modeData.detectFreeCell.minDensity;
            detectFreeCell._restrictToColor = nodeTO->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor))
                : std::nullopt;
            sensorDesc._mode = detectFreeCell;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectCreature) {
            DetectCreatureGenomeDesc detectCreature;
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
    case CellType_Generator: {
        GeneratorGenomeDesc generatorDesc;
        generatorDesc._autoTriggerInterval = nodeTO->cellTypeData.generator.autoTriggerInterval;
        generatorDesc._pulseType = nodeTO->cellTypeData.generator.pulseType;
        generatorDesc._alternationInterval = nodeTO->cellTypeData.generator.alternationInterval;
        nodeDesc._cellType = generatorDesc;
    } break;
    case CellType_Attacker: {
        AttackerGenomeDesc attackerDesc;
        if (nodeTO->cellTypeData.attacker.mode == AttackerMode_FreeCell) {
            AttackFreeCellGenomeDesc attackFreeCell;
            attackFreeCell._restrictToColor = nodeTO->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor))
                : std::nullopt;
            attackerDesc._mode = attackFreeCell;
        } else if (nodeTO->cellTypeData.attacker.mode == AttackerMode_Creature) {
            AttackCreatureGenomeDesc attackCreature;
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
    case CellType_Injector: {
        InjectorGenomeDesc injectorDesc;
        injectorDesc._geneIndex = nodeTO->cellTypeData.injector.geneIndex;
        nodeDesc._cellType = injectorDesc;
    } break;
    case CellType_Muscle: {
        MuscleGenomeDesc muscleDesc;
        switch (nodeTO->cellTypeData.muscle.mode) {
        case MuscleMode_AutoBending: {
            AutoBendingGenomeDesc bendingDesc;
            bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
            bendingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.autoBending.forwardBackwardRatio;
            muscleDesc._mode = bendingDesc;
        } break;
        case MuscleMode_ManualBending: {
            ManualBendingGenomeDesc bendingDesc;
            bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
            bendingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
            muscleDesc._mode = bendingDesc;
        } break;
        case MuscleMode_AngleBending: {
            AngleBendingGenomeDesc bendingDesc;
            bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
            bendingDesc._attractionRepulsionRatio = nodeTO->cellTypeData.muscle.modeData.angleBending.attractionRepulsionRatio;
            muscleDesc._mode = bendingDesc;
        } break;
        case MuscleMode_AutoCrawling: {
            AutoCrawlingGenomeDesc crawlingDesc;
            crawlingDesc._maxDistanceDeviation = nodeTO->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
            crawlingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.autoCrawling.forwardBackwardRatio;
            muscleDesc._mode = crawlingDesc;
        } break;
        case MuscleMode_ManualCrawling: {
            ManualCrawlingGenomeDesc crawlingDesc;
            crawlingDesc._maxDistanceDeviation = nodeTO->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
            crawlingDesc._forwardBackwardRatio = nodeTO->cellTypeData.muscle.modeData.manualCrawling.forwardBackwardRatio;
            muscleDesc._mode = crawlingDesc;
        } break;
        case MuscleMode_DirectMovement: {
            DirectMovementGenomeDesc movementDesc;
            muscleDesc._mode = movementDesc;
        } break;
        }
        nodeDesc._cellType = muscleDesc;
    } break;
    case CellType_Defender: {
        DefenderGenomeDesc defenderDesc;
        defenderDesc._mode = nodeTO->cellTypeData.defender.mode;
        nodeDesc._cellType = defenderDesc;
    } break;
    case CellType_Reconnector: {
        ReconnectorGenomeDesc reconnectorDesc;
        if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_Structure) {
            ReconnectStructureGenomeDesc reconnectStructure;
            reconnectorDesc._mode = reconnectStructure;
        } else if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
            ReconnectFreeCellGenomeDesc reconnectFreeCell;
            reconnectFreeCell._restrictToColor = nodeTO->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor != 255
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColor))
                : std::nullopt;
            reconnectorDesc._mode = reconnectFreeCell;
        } else if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
            ReconnectCreatureGenomeDesc reconnectCreature;
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
    case CellType_Detonator: {
        DetonatorGenomeDesc detonatorDesc;
        detonatorDesc._countdown = nodeTO->cellTypeData.detonator.countdown;
        nodeDesc._cellType = detonatorDesc;
    } break;
    case CellType_Digestor: {
        DigestorGenomeDesc digestorDesc;
        digestorDesc._rawEnergyConductivity = nodeTO->cellTypeData.digestor.rawEnergyConductivity;
        nodeDesc._cellType = digestorDesc;
    } break;
    case CellType_Memory: {
        MemoryGenomeDesc memoryDesc;
        auto const& memoryTO = nodeTO->cellTypeData.memory;
        if (memoryTO.mode == MemoryMode_SignalDelay) {
            SignalDelayGenomeDesc signalDelay;
            signalDelay._delay = memoryTO.modeData.signalDelay.delay;
            memoryDesc._mode = signalDelay;
        } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
            SignalRecorderGenomeDesc signalRecorder;
            signalRecorder._readOnly = memoryTO.modeData.signalRecorder.readOnly;
            signalRecorder._numWrittenSignalEntries = memoryTO.modeData.signalRecorder.numWrittenSignalEntries;
            memoryDesc._mode = signalRecorder;
        } else if (memoryTO.mode == MemoryMode_SignalStorage) {
            SignalStorageGenomeDesc signalStorage;
            signalStorage._readOnly = memoryTO.modeData.signalStorage.readOnly;
            memoryDesc._mode = signalStorage;
        } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
            SignalIntegratorGenomeDesc signalIntegrator;
            signalIntegrator._newSignalWeight = memoryTO.modeData.signalIntegrator.newSignalWeight;
            memoryDesc._mode = signalIntegrator;
        }
        memoryDesc._channelBitMask = memoryTO.channelBitMask;
        auto const& signalEntriesTO = getFromHeap<SignalEntryGenomeTO>(to.heap, memoryTO.signalEntriesDataIndex);
        copyMemoryEntriesToDescription(memoryDesc._signalEntries, signalEntriesTO, memoryTO.numSignalEntries);
        nodeDesc._cellType = memoryDesc;
    } break;
    case CellType_Communicator: {
        CommunicatorGenomeDesc communicatorDesc;
        auto const& communicatorTO = nodeTO->cellTypeData.communicator;
        if (communicatorTO.mode == CommunicatorMode_Sender) {
            SenderGenomeDesc sender;
            sender._range = communicatorTO.modeData.sender.range;
            sender._maxTimesSent = communicatorTO.modeData.sender.maxTimesSent;
            communicatorDesc._mode = sender;
        } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
            ReceiverGenomeDesc receiver;
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

GenomeDesc DescriptionConverterService::createGenomeDesc(TOs const& to, int genomeIndex) const
{
    auto const& genomeTO = to.genomes[genomeIndex];

    GenomeDesc result;
    result._id = genomeTO.id;
    NumberGenerator::get().adaptMaxIds({.entityId = genomeTO.id});
    result._name = char64ToString(genomeTO.name);
    result._frontAngle = genomeTO.frontAngle;
    result._genes.reserve(genomeTO.numGenes);

    CHECK(genomeTO.geneArrayIndex + genomeTO.numGenes <= *to.numGenes);
    for (int j = 0; j < genomeTO.numGenes; ++j) {
        auto geneTO = to.genes + genomeTO.geneArrayIndex + j;

        GeneDesc geneDesc;
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
            geneDesc._nodes.emplace_back(createNodeDesc(to, nodeTO));
        }

        result._genes.emplace_back(geneDesc);
    }
    return result;
}

CreatureDesc DescriptionConverterService::createCreatureDesc(TOs const& to, int creatureIndex) const
{
    CreatureDesc result;

    auto const& creatureTO = to.creatures[creatureIndex];
    result._id = creatureTO.id;
    NumberGenerator::get().adaptMaxIds({.entityId = creatureTO.id});
    result._ancestorId = creatureTO.ancestorId != VALUE_NOT_SET_UINT64 ? std::make_optional(creatureTO.ancestorId) : std::nullopt;
    result._generation = creatureTO.generation;
    result._lineageId = creatureTO.lineageId;
    NumberGenerator::get().adaptMaxIds({.entityId = creatureTO.lineageId});
    result._numObjects = creatureTO.numObjects;
    result._frontAngleId = creatureTO.frontAngleId;

    return result;
}

EnergyDesc DescriptionConverterService::createEnergyDesc(TOs const& to, int particleIndex) const
{
    auto const& energyParticle = to.energyParticles[particleIndex];
    NumberGenerator::get().adaptMaxIds({.entityId = energyParticle.id});
    return EnergyDesc()
        .id(energyParticle.id)
        .pos({energyParticle.pos.x, energyParticle.pos.y})
        .vel({energyParticle.vel.x, energyParticle.vel.y})
        .energy(energyParticle.energy)
        .color(energyParticle.color);
}

void DescriptionConverterService::convertGenomeToTO(
    std::vector<GenomeTO>& genomeTOs,
    std::vector<GeneTO>& geneTOs,
    std::vector<NodeTO>& nodeTOs,
    std::vector<uint8_t>& heap,
    GenomeDesc const& genome,
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
            case CellType_Base: {
            } break;
            case CellType_Depot: {
                auto const& depotDesc = std::get<DepotGenomeDesc>(nodeDesc._cellType);
                auto& depotTO = nodeTO.cellTypeData.depot;
                depotTO.storageLimit = depotDesc._storageLimit;
                depotTO.initialStoredUsableEnergy = depotDesc._initialStoredUsableEnergy;
            } break;
            case CellType_Constructor: {
                auto const& constructorDesc = std::get<ConstructorGenomeDesc>(nodeDesc._cellType);
                auto& constructorTO = nodeTO.cellTypeData.constructor;
                constructorTO.autoTriggerInterval = static_cast<uint32_t>(constructorDesc._autoTriggerInterval.value_or(0));
                constructorTO.geneIndex = constructorDesc._geneIndex;
                constructorTO.constructionActivationTime = constructorDesc._constructionActivationTime;
                constructorTO.constructionAngle = constructorDesc._constructionAngle;
                constructorTO.provideEnergy = constructorDesc._provideEnergy;
            } break;
            case CellType_Sensor: {
                auto const& sensorDesc = std::get<SensorGenomeDesc>(nodeDesc._cellType);
                auto& sensorTO = nodeTO.cellTypeData.sensor;
                sensorTO.autoTriggerInterval = static_cast<uint32_t>(sensorDesc._autoTriggerInterval.value_or(0));
                sensorTO.minRange = static_cast<uint16_t>(sensorDesc._minRange);
                sensorTO.maxRange = static_cast<uint16_t>(sensorDesc._maxRange);
                sensorTO.mode = sensorDesc.getMode();

                if (sensorTO.mode == SensorMode_Telemetry) {
                    //auto const& telemetryDesc = std::get<TelemetryGenomeDesc>(sensorDesc._mode);
                    //auto& telemetryTO = sensorTO.modeData.telemetry;
                } else if (sensorTO.mode == SensorMode_DetectEnergy) {
                    auto const& detectEnergyDesc = std::get<DetectEnergyGenomeDesc>(sensorDesc._mode);
                    auto& detectEnergyTO = sensorTO.modeData.detectEnergy;
                    detectEnergyTO.minDensity = detectEnergyDesc._minDensity;
                } else if (sensorTO.mode == SensorMode_DetectStructure) {
                } else if (sensorTO.mode == SensorMode_DetectFreeCell) {
                    auto const& detectFreeCellDesc = std::get<DetectFreeCellGenomeDesc>(sensorDesc._mode);
                    auto& detectFreeCellTO = sensorTO.modeData.detectFreeCell;
                    detectFreeCellTO.minDensity = detectFreeCellDesc._minDensity;
                    detectFreeCellTO.restrictToColor = static_cast<uint8_t>(detectFreeCellDesc._restrictToColor.value_or(255));
                } else if (sensorTO.mode == SensorMode_DetectCreature) {
                    auto const& detectCreatureDesc = std::get<DetectCreatureGenomeDesc>(sensorDesc._mode);
                    auto& detectCreatureTO = sensorTO.modeData.detectCreature;
                    detectCreatureTO.minNumCells = static_cast<uint32_t>(detectCreatureDesc._minNumCells.value_or(0));
                    detectCreatureTO.maxNumCells = static_cast<uint32_t>(detectCreatureDesc._maxNumCells.value_or(0));
                    detectCreatureTO.restrictToColor = static_cast<uint8_t>(detectCreatureDesc._restrictToColor.value_or(255));
                    detectCreatureTO.restrictToLineage = detectCreatureDesc._restrictToLineage;
                }
            } break;
            case CellType_Generator: {
                auto const& generatorDesc = std::get<GeneratorGenomeDesc>(nodeDesc._cellType);
                auto& generatorTO = nodeTO.cellTypeData.generator;
                generatorTO.autoTriggerInterval = generatorDesc._autoTriggerInterval;
                generatorTO.pulseType = generatorDesc._pulseType;
                generatorTO.alternationInterval = generatorDesc._alternationInterval;
            } break;
            case CellType_Attacker: {
                auto const& attackerDesc = std::get<AttackerGenomeDesc>(nodeDesc._cellType);
                auto& attackerTO = nodeTO.cellTypeData.attacker;
                attackerTO.mode = attackerDesc.getMode();
                if (attackerTO.mode == AttackerMode_FreeCell) {
                    auto const& attackFreeCellDesc = std::get<AttackFreeCellGenomeDesc>(attackerDesc._mode);
                    auto& attackFreeCellTO = attackerTO.modeData.attackFreeCell;
                    attackFreeCellTO.restrictToColor = static_cast<uint8_t>(attackFreeCellDesc._restrictToColor.value_or(255));
                } else if (attackerTO.mode == AttackerMode_Creature) {
                    auto const& attackCreatureDesc = std::get<AttackCreatureGenomeDesc>(attackerDesc._mode);
                    auto& attackCreatureTO = attackerTO.modeData.attackCreature;
                    attackCreatureTO.minNumCells = static_cast<uint32_t>(attackCreatureDesc._minNumCells.value_or(0));
                    attackCreatureTO.maxNumCells = static_cast<uint32_t>(attackCreatureDesc._maxNumCells.value_or(0));
                    attackCreatureTO.restrictToColor = static_cast<uint8_t>(attackCreatureDesc._restrictToColor.value_or(255));
                    attackCreatureTO.restrictToLineage = attackCreatureDesc._restrictToLineage;
                }
            } break;
            case CellType_Injector: {
                auto const& injectorDesc = std::get<InjectorGenomeDesc>(nodeDesc._cellType);
                auto& injectorTO = nodeTO.cellTypeData.injector;
                injectorTO.geneIndex = injectorDesc._geneIndex;
            } break;
            case CellType_Muscle: {
                auto const& muscleDesc = std::get<MuscleGenomeDesc>(nodeDesc._cellType);
                auto& muscleTO = nodeTO.cellTypeData.muscle;
                muscleTO.mode = muscleDesc.getMode();
                switch (muscleDesc.getMode()) {
                case MuscleMode_AutoBending: {
                    auto const& autoBendingDesc = std::get<AutoBendingGenomeDesc>(muscleDesc._mode);
                    auto& autoBendingTO = muscleTO.modeData.autoBending;
                    autoBendingTO.maxAngleDeviation = autoBendingDesc._maxAngleDeviation;
                    autoBendingTO.forwardBackwardRatio = autoBendingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_ManualBending: {
                    auto const& manualBendingDesc = std::get<ManualBendingGenomeDesc>(muscleDesc._mode);
                    auto& manualBendingTO = muscleTO.modeData.manualBending;
                    manualBendingTO.maxAngleDeviation = manualBendingDesc._maxAngleDeviation;
                    manualBendingTO.forwardBackwardRatio = manualBendingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_AngleBending: {
                    auto const& angleBendingDesc = std::get<AngleBendingGenomeDesc>(muscleDesc._mode);
                    auto& angleBendingTO = muscleTO.modeData.angleBending;
                    angleBendingTO.maxAngleDeviation = angleBendingDesc._maxAngleDeviation;
                    angleBendingTO.attractionRepulsionRatio = angleBendingDesc._attractionRepulsionRatio;
                } break;
                case MuscleMode_AutoCrawling: {
                    auto const& autoCrawlingDesc = std::get<AutoCrawlingGenomeDesc>(muscleDesc._mode);
                    auto& autoCrawlingTO = muscleTO.modeData.autoCrawling;
                    autoCrawlingTO.maxDistanceDeviation = autoCrawlingDesc._maxDistanceDeviation;
                    autoCrawlingTO.forwardBackwardRatio = autoCrawlingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_ManualCrawling: {
                    auto const& manualCrawlingDesc = std::get<ManualCrawlingGenomeDesc>(muscleDesc._mode);
                    auto& manualCrawlingTO = muscleTO.modeData.manualCrawling;
                    manualCrawlingTO.maxDistanceDeviation = manualCrawlingDesc._maxDistanceDeviation;
                    manualCrawlingTO.forwardBackwardRatio = manualCrawlingDesc._forwardBackwardRatio;
                } break;
                case MuscleMode_DirectMovement: {
                } break;
                }
            } break;
            case CellType_Defender: {
                auto const& defenderDesc = std::get<DefenderGenomeDesc>(nodeDesc._cellType);
                auto& defenderTO = nodeTO.cellTypeData.defender;
                defenderTO.mode = defenderDesc._mode;
            } break;
            case CellType_Reconnector: {
                auto const& reconnectorDesc = std::get<ReconnectorGenomeDesc>(nodeDesc._cellType);
                auto& reconnectorTO = nodeTO.cellTypeData.reconnector;
                reconnectorTO.mode = reconnectorDesc.getMode();
                if (reconnectorTO.mode == ReconnectorMode_Structure) {
                    // No data to copy
                } else if (reconnectorTO.mode == ReconnectorMode_FreeCell) {
                    auto const& reconnectFreeCellDesc = std::get<ReconnectFreeCellGenomeDesc>(reconnectorDesc._mode);
                    auto& reconnectFreeCellTO = reconnectorTO.modeData.reconnectFreeCell;
                    reconnectFreeCellTO.restrictToColor = static_cast<uint8_t>(reconnectFreeCellDesc._restrictToColor.value_or(255));
                } else if (reconnectorTO.mode == ReconnectorMode_Creature) {
                    auto const& reconnectCreatureDesc = std::get<ReconnectCreatureGenomeDesc>(reconnectorDesc._mode);
                    auto& reconnectCreatureTO = reconnectorTO.modeData.reconnectCreature;
                    reconnectCreatureTO.minNumCells = static_cast<uint32_t>(reconnectCreatureDesc._minNumCells.value_or(0));
                    reconnectCreatureTO.maxNumCells = static_cast<uint32_t>(reconnectCreatureDesc._maxNumCells.value_or(0));
                    reconnectCreatureTO.restrictToColor = static_cast<uint8_t>(reconnectCreatureDesc._restrictToColor.value_or(255));
                    reconnectCreatureTO.restrictToLineage = reconnectCreatureDesc._restrictToLineage;
                }
            } break;
            case CellType_Detonator: {
                auto const& detonatorDesc = std::get<DetonatorGenomeDesc>(nodeDesc._cellType);
                auto& detonatorTO = nodeTO.cellTypeData.detonator;
                detonatorTO.countdown = detonatorDesc._countdown;
            } break;
            case CellType_Digestor: {
                auto const& digestorDesc = std::get<DigestorGenomeDesc>(nodeDesc._cellType);
                auto& digestorTO = nodeTO.cellTypeData.digestor;
                digestorTO.rawEnergyConductivity = digestorDesc._rawEnergyConductivity;
            } break;
            case CellType_Memory: {
                auto const& memoryDesc = std::get<MemoryGenomeDesc>(nodeDesc._cellType);
                auto& memoryTO = nodeTO.cellTypeData.memory;
                memoryTO.mode = memoryDesc.getMode();
                if (memoryTO.mode == MemoryMode_SignalDelay) {
                    auto const& signalDelayDesc = std::get<SignalDelayGenomeDesc>(memoryDesc._mode);
                    memoryTO.modeData.signalDelay.delay = signalDelayDesc._delay;
                } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
                    auto const& signalRecorderDesc = std::get<SignalRecorderGenomeDesc>(memoryDesc._mode);
                    memoryTO.modeData.signalRecorder.readOnly = signalRecorderDesc._readOnly;
                    memoryTO.modeData.signalRecorder.numWrittenSignalEntries = signalRecorderDesc._numWrittenSignalEntries;
                } else if (memoryTO.mode == MemoryMode_SignalStorage) {
                    auto const& signalStorageDesc = std::get<SignalStorageGenomeDesc>(memoryDesc._mode);
                    memoryTO.modeData.signalStorage.readOnly = signalStorageDesc._readOnly;
                } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
                    auto const& signalIntegratorDesc = std::get<SignalIntegratorGenomeDesc>(memoryDesc._mode);
                    memoryTO.modeData.signalIntegrator.newSignalWeight = signalIntegratorDesc._newSignalWeight;
                }
                memoryTO.channelBitMask = memoryDesc._channelBitMask;
                memoryTO.numSignalEntries = toInt(memoryDesc._signalEntries.size());
                memoryTO.signalEntriesDataIndex = heap.size();
                heap.resize(heap.size() + sizeof(SignalEntryGenomeTO) * memoryTO.numSignalEntries);
                auto signalEntriesTO = reinterpret_cast<SignalEntryGenomeTO*>(heap.data() + memoryTO.signalEntriesDataIndex);
                copyMemoryEntriesToTO(signalEntriesTO, memoryDesc._signalEntries);
            } break;
            case CellType_Communicator: {
                auto const& communicatorDesc = std::get<CommunicatorGenomeDesc>(nodeDesc._cellType);
                auto& communicatorTO = nodeTO.cellTypeData.communicator;
                communicatorTO.mode = communicatorDesc.getMode();
                if (communicatorTO.mode == CommunicatorMode_Sender) {
                    auto const& senderDesc = std::get<SenderGenomeDesc>(communicatorDesc._mode);
                    communicatorTO.modeData.sender.range = senderDesc._range;
                    communicatorTO.modeData.sender.maxTimesSent = senderDesc._maxTimesSent;
                } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
                    auto const& receiverDesc = std::get<ReceiverGenomeDesc>(communicatorDesc._mode);
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
    CreatureDesc const& creatureDesc,
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
    creatureTO.numObjects = creatureDesc._numObjects;
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

void DescriptionConverterService::convertObjectToTO(
    std::vector<ObjectTO>& objectTOs,
    std::vector<uint8_t>& heap,
    std::unordered_map<uint64_t, uint64_t>& objectTOIndexById,
    ObjectDesc const& objectDesc,
    std::unordered_map<uint64_t, uint64_t> const& creatureTOIndexById) const
{
    auto objectIndex = objectTOs.size();
    objectTOs.resize(objectIndex + 1);

    ObjectTO& objectTO = objectTOs.at(objectIndex);
    objectTO.id = objectDesc._id;
    objectTOIndexById.insert_or_assign(objectTO.id, objectIndex);

    objectTO.pos = {objectDesc._pos.x, objectDesc._pos.y};
    objectTO.vel = {objectDesc._vel.x, objectDesc._vel.y};
    objectTO.stiffness = objectDesc._stiffness;
    objectTO.numConnections = 0;
    objectTO.fixed = objectDesc._fixed;
    objectTO.sticky = objectDesc._sticky;
    objectTO.color = objectDesc._color;

    // Set object type
    objectTO.type = objectDesc.getObjectType();

    // Handle Structure and FreeCell object types
    if (objectTO.type == ObjectType_Structure) {
        StructureDesc const& structureDesc = objectDesc.getStructureRef();
        objectTO.typeData.structure.energy = structureDesc._energy;
    } else if (objectTO.type == ObjectType_FreeCell) {
        FreeCellDesc const& freeCellDesc = objectDesc.getFreeCellRef();
        objectTO.typeData.freeCell.energy = freeCellDesc._energy;
        objectTO.typeData.freeCell.age = freeCellDesc._age;
    } else if (objectTO.type == ObjectType_Cell) {

        // ObjectType_Cell - access cell data
        CellDesc const& cellDesc = objectDesc.getCellRef();

        objectTO.typeData.cell.creatureIndex = creatureTOIndexById.at(cellDesc._creatureId.value());
        objectTO.typeData.cell.usableEnergy = cellDesc._usableEnergy;
        checkAndCorrectInvalidEnergy(objectTO.typeData.cell.usableEnergy);
        objectTO.typeData.cell.rawEnergy = cellDesc._rawEnergy;
        objectTO.typeData.cell.cellState = cellDesc._cellState;
        objectTO.typeData.cell.cellType = cellDesc.getCellType();
        objectTO.typeData.cell.nodeIndex = cellDesc._nodeIndex;
        objectTO.typeData.cell.parentNodeIndex = cellDesc._parentNodeIndex;
        objectTO.typeData.cell.geneIndex = cellDesc._geneIndex;
        objectTO.typeData.cell.frontAngle = cellDesc._frontAngle.value_or(VALUE_NOT_SET_FLOAT);
        objectTO.typeData.cell.frontAngleId = cellDesc._frontAngleId;
        objectTO.typeData.cell.headCell = cellDesc._headCell;
        objectTO.typeData.cell.age = cellDesc._age;
        objectTO.typeData.cell.activationTime = cellDesc._activationTime;
        objectTO.typeData.cell.event = cellDesc._event;
        objectTO.typeData.cell.eventCounter = cellDesc._eventCounter;
        objectTO.typeData.cell.eventPos = {cellDesc._eventPos.x, cellDesc._eventPos.y};

        objectTO.typeData.cell.neuralNetworkDataIndex = heap.size();
        heap.resize(heap.size() + sizeof(NeuralNetworkTO));
        auto neuralNetworkTO = reinterpret_cast<NeuralNetworkTO*>(heap.data() + heap.size() - sizeof(NeuralNetworkTO));
        *neuralNetworkTO = convert(cellDesc._neuralNetwork);
        auto cellType = cellDesc.getCellType();

        switch (cellType) {
        case CellType_Base: {
            BaseTO baseTO;
            objectTO.typeData.cell.cellTypeData.base = baseTO;
        } break;
        case CellType_Depot: {
            auto const& depotDesc = std::get<DepotDesc>(cellDesc._cellType);
            DepotTO& depotTO = objectTO.typeData.cell.cellTypeData.depot;
            depotTO.storageLimit = depotDesc._storageLimit;
            depotTO.storedUsableEnergy = depotDesc._storedUsableEnergy;
        } break;
        case CellType_Constructor: {
            auto const& constructorDesc = std::get<ConstructorDesc>(cellDesc._cellType);
            ConstructorTO& constructorTO = objectTO.typeData.cell.cellTypeData.constructor;
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
            auto const& sensorDesc = std::get<SensorDesc>(cellDesc._cellType);
            SensorTO& sensorTO = objectTO.typeData.cell.cellTypeData.sensor;
            sensorTO.autoTriggerInterval = static_cast<uint32_t>(sensorDesc._autoTriggerInterval.value_or(0));
            sensorTO.minRange = static_cast<uint16_t>(sensorDesc._minRange);
            sensorTO.maxRange = static_cast<uint16_t>(sensorDesc._maxRange);
            sensorTO.mode = sensorDesc.getMode();

            if (sensorTO.mode == SensorMode_Telemetry) {
                //auto const& telemetryDesc = std::get<TelemetryDesc>(sensorDesc._mode);
                //TelemetryTO& telemetryTO = sensorTO.modeData.telemetry;
            } else if (sensorTO.mode == SensorMode_DetectEnergy) {
                auto const& detectEnergyDesc = std::get<DetectEnergyDesc>(sensorDesc._mode);
                DetectEnergyTO& detectEnergyTO = sensorTO.modeData.detectEnergy;
                detectEnergyTO.minDensity = detectEnergyDesc._minDensity;
            } else if (sensorTO.mode == SensorMode_DetectStructure) {
            } else if (sensorTO.mode == SensorMode_DetectFreeCell) {
                auto const& detectFreeCellDesc = std::get<DetectFreeCellDesc>(sensorDesc._mode);
                DetectFreeCellTO& detectFreeCellTO = sensorTO.modeData.detectFreeCell;
                detectFreeCellTO.minDensity = detectFreeCellDesc._minDensity;
                detectFreeCellTO.restrictToColor = static_cast<uint8_t>(detectFreeCellDesc._restrictToColor.value_or(255));
            } else if (sensorTO.mode == SensorMode_DetectCreature) {
                auto const& detectCreatureDesc = std::get<DetectCreatureDesc>(sensorDesc._mode);
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
            auto const& generatorDesc = std::get<GeneratorDesc>(cellDesc._cellType);
            GeneratorTO& generatorTO = objectTO.typeData.cell.cellTypeData.generator;
            generatorTO.autoTriggerInterval = generatorDesc._autoTriggerInterval;
            generatorTO.pulseType = generatorDesc._pulseType;
            generatorTO.alternationInterval = generatorDesc._alternationInterval;
            generatorTO.numPulses = generatorDesc._numPulses;
        } break;
        case CellType_Attacker: {
            auto const& attackerDesc = std::get<AttackerDesc>(cellDesc._cellType);
            AttackerTO& attackerTO = objectTO.typeData.cell.cellTypeData.attacker;
            attackerTO.mode = attackerDesc.getMode();
            if (attackerTO.mode == AttackerMode_FreeCell) {
                auto const& attackFreeCellDesc = std::get<AttackFreeCellDesc>(attackerDesc._mode);
                attackerTO.modeData.attackFreeCell.restrictToColor = static_cast<uint8_t>(attackFreeCellDesc._restrictToColor.value_or(255));
            } else if (attackerTO.mode == AttackerMode_Creature) {
                auto const& attackCreatureDesc = std::get<AttackCreatureDesc>(attackerDesc._mode);
                attackerTO.modeData.attackCreature.minNumCells = static_cast<uint32_t>(attackCreatureDesc._minNumCells.value_or(0));
                attackerTO.modeData.attackCreature.maxNumCells = static_cast<uint32_t>(attackCreatureDesc._maxNumCells.value_or(0));
                attackerTO.modeData.attackCreature.restrictToColor = static_cast<uint8_t>(attackCreatureDesc._restrictToColor.value_or(255));
                attackerTO.modeData.attackCreature.restrictToLineage = attackCreatureDesc._restrictToLineage;
            }
        } break;
        case CellType_Injector: {
            auto const& injectorDesc = std::get<InjectorDesc>(cellDesc._cellType);
            InjectorTO& injectorTO = objectTO.typeData.cell.cellTypeData.injector;
            injectorTO.geneIndex = static_cast<uint16_t>(injectorDesc._geneIndex);
        } break;
        case CellType_Muscle: {
            auto const& muscleDesc = std::get<MuscleDesc>(cellDesc._cellType);
            MuscleTO& muscleTO = objectTO.typeData.cell.cellTypeData.muscle;
            muscleTO.mode = muscleDesc.getMode();
            if (muscleTO.mode == MuscleMode_AutoBending) {
                auto const& bendingDesc = std::get<AutoBendingDesc>(muscleDesc._mode);
                AutoBendingTO& bendingTO = muscleTO.modeData.autoBending;
                bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
                bendingTO.forwardBackwardRatio = bendingDesc._forwardBackwardRatio;
                bendingTO.initialAngle = bendingDesc._initialAngle.value_or(VALUE_NOT_SET_FLOAT);
                bendingTO.forward = bendingDesc._forward;
                bendingTO.activation = bendingDesc._activation;
                bendingTO.activationCountdown = bendingDesc._activationCountdown;
                bendingTO.impulseAlreadyApplied = bendingDesc._impulseAlreadyApplied;
            } else if (muscleTO.mode == MuscleMode_ManualBending) {
                auto const& bendingDesc = std::get<ManualBendingDesc>(muscleDesc._mode);
                ManualBendingTO& bendingTO = muscleTO.modeData.manualBending;
                bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
                bendingTO.forwardBackwardRatio = bendingDesc._forwardBackwardRatio;
                bendingTO.initialAngle = bendingDesc._initialAngle.value_or(VALUE_NOT_SET_FLOAT);
                bendingTO.lastAngleDelta = bendingDesc._lastAngleDelta;
                bendingTO.impulseAlreadyApplied = bendingDesc._impulseAlreadyApplied;
            } else if (muscleTO.mode == MuscleMode_AngleBending) {
                auto const& bendingDesc = std::get<AngleBendingDesc>(muscleDesc._mode);
                AngleBendingTO& bendingTO = muscleTO.modeData.angleBending;
                bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
                bendingTO.attractionRepulsionRatio = bendingDesc._attractionRepulsionRatio;
                bendingTO.initialAngle = bendingDesc._initialAngle.value_or(VALUE_NOT_SET_FLOAT);
            } else if (muscleTO.mode == MuscleMode_AutoCrawling) {
                auto const& crawlingDesc = std::get<AutoCrawlingDesc>(muscleDesc._mode);
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
                auto const& crawlingDesc = std::get<ManualCrawlingDesc>(muscleDesc._mode);
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
            auto const& defenderDesc = std::get<DefenderDesc>(cellDesc._cellType);
            DefenderTO& defenderTO = objectTO.typeData.cell.cellTypeData.defender;
            defenderTO.mode = defenderDesc._mode;
        } break;
        case CellType_Reconnector: {
            auto const& reconnectorDesc = std::get<ReconnectorDesc>(cellDesc._cellType);
            ReconnectorTO& reconnectorTO = objectTO.typeData.cell.cellTypeData.reconnector;
            reconnectorTO.mode = reconnectorDesc.getMode();
            if (reconnectorTO.mode == ReconnectorMode_Structure) {
                // No data to copy
            } else if (reconnectorTO.mode == ReconnectorMode_FreeCell) {
                auto const& reconnectFreeCellDesc = std::get<ReconnectFreeCellDesc>(reconnectorDesc._mode);
                ReconnectFreeCellTO& reconnectFreeCellTO = reconnectorTO.modeData.reconnectFreeCell;
                reconnectFreeCellTO.restrictToColor = static_cast<uint8_t>(reconnectFreeCellDesc._restrictToColor.value_or(255));
            } else if (reconnectorTO.mode == ReconnectorMode_Creature) {
                auto const& reconnectCreatureDesc = std::get<ReconnectCreatureDesc>(reconnectorDesc._mode);
                ReconnectCreatureTO& reconnectCreatureTO = reconnectorTO.modeData.reconnectCreature;
                reconnectCreatureTO.minNumCells = static_cast<uint32_t>(reconnectCreatureDesc._minNumCells.value_or(0));
                reconnectCreatureTO.maxNumCells = static_cast<uint32_t>(reconnectCreatureDesc._maxNumCells.value_or(0));
                reconnectCreatureTO.restrictToColor = static_cast<uint8_t>(reconnectCreatureDesc._restrictToColor.value_or(255));
                reconnectCreatureTO.restrictToLineage = reconnectCreatureDesc._restrictToLineage;
            }
        } break;
        case CellType_Detonator: {
            auto const& detonatorDesc = std::get<DetonatorDesc>(cellDesc._cellType);
            DetonatorTO& detonatorTO = objectTO.typeData.cell.cellTypeData.detonator;
            detonatorTO.state = detonatorDesc._state;
            detonatorTO.countdown = detonatorDesc._countdown;
        } break;
        case CellType_Digestor: {
            auto const& digestorDesc = std::get<DigestorDesc>(cellDesc._cellType);
            DigestorTO& digestorTO = objectTO.typeData.cell.cellTypeData.digestor;
            digestorTO.rawEnergyConductivity = digestorDesc._rawEnergyConductivity;
        } break;
        case CellType_Memory: {
            auto const& memoryDesc = std::get<MemoryDesc>(cellDesc._cellType);
            auto& memoryTO = objectTO.typeData.cell.cellTypeData.memory;
            memoryTO.mode = memoryDesc.getMode();
            if (memoryTO.mode == MemoryMode_SignalDelay) {
                auto const& signalDelayDesc = std::get<SignalDelayDesc>(memoryDesc._mode);
                memoryTO.modeData.signalDelay.delay = signalDelayDesc._delay;
                memoryTO.modeData.signalDelay.numSignalEntriesInitialized = signalDelayDesc._numSignalEntriesInitialized;
                memoryTO.modeData.signalDelay.ringBufferIndex = signalDelayDesc._ringBufferIndex;
            } else if (memoryTO.mode == MemoryMode_SignalRecorder) {
                auto const& signalRecorderDesc = std::get<SignalRecorderDesc>(memoryDesc._mode);
                memoryTO.modeData.signalRecorder.readOnly = signalRecorderDesc._readOnly;
                memoryTO.modeData.signalRecorder.state = signalRecorderDesc._state;
                memoryTO.modeData.signalRecorder.numWrittenSignalEntries = signalRecorderDesc._numWrittenSignalEntries;
                memoryTO.modeData.signalRecorder.numReadSignalEntries = signalRecorderDesc._numReadSignalEntries;
            } else if (memoryTO.mode == MemoryMode_SignalStorage) {
                auto const& signalStorageDesc = std::get<SignalStorageDesc>(memoryDesc._mode);
                memoryTO.modeData.signalStorage.readOnly = signalStorageDesc._readOnly;
            } else if (memoryTO.mode == MemoryMode_SignalIntegrator) {
                auto const& signalIntegratorDesc = std::get<SignalIntegratorDesc>(memoryDesc._mode);
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
            auto const& communicatorDesc = std::get<CommunicatorDesc>(cellDesc._cellType);
            CommunicatorTO& communicatorTO = objectTO.typeData.cell.cellTypeData.communicator;
            communicatorTO.mode = communicatorDesc.getMode();
            if (communicatorTO.mode == CommunicatorMode_Sender) {
                auto const& senderDesc = std::get<SenderDesc>(communicatorDesc._mode);
                communicatorTO.modeData.sender.range = senderDesc._range;
                communicatorTO.modeData.sender.maxTimesSent = senderDesc._maxTimesSent;
            } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
                auto const& receiverDesc = std::get<ReceiverDesc>(communicatorDesc._mode);
                communicatorTO.modeData.receiver.restrictToColor = static_cast<uint8_t>(receiverDesc._restrictToColor.value_or(255));
                communicatorTO.modeData.receiver.restrictToLineage = receiverDesc._restrictToLineage;
            }
        } break;
        }
        objectTO.typeData.cell.signalRestriction.mode = cellDesc._signalRestriction._mode;
        objectTO.typeData.cell.signalRestriction.baseAngle = cellDesc._signalRestriction._baseAngle;
        objectTO.typeData.cell.signalRestriction.openingAngle = cellDesc._signalRestriction._openingAngle;
        objectTO.typeData.cell.signalState = cellDesc._signalState;
        if (cellDesc._signalState == SignalState_Active) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                objectTO.typeData.cell.signal.channels[i] = cellDesc._signal._channels[i];
            }
            objectTO.typeData.cell.signal.numTimesSent = cellDesc._signal._numTimesSent;
        }
    } else {
        CHECK(false);
    }
}

void DescriptionConverterService::addParticle(std::vector<EnergyTO>& particleTOs, EnergyDesc const& particleDesc) const
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
    std::vector<ObjectTO>& objectTOs,
    ObjectDesc const& cellToAdd,
    std::unordered_map<uint64_t, uint64_t> const& objectIndexByIds) const
{
    int index = 0;
    auto& objectTO = objectTOs.at(objectIndexByIds.at(cellToAdd._id));
    float angleOffset = 0;
    for (ConnectionDesc const& connection : cellToAdd._connections) {
        objectTO.connections[index].objectIndex = objectIndexByIds.at(connection._objectId);
        objectTO.connections[index].distance = connection._distance;
        objectTO.connections[index].angleFromPrevious = connection._angleFromPrevious + angleOffset;
        ++index;
        angleOffset = 0;
    }
    if (angleOffset != 0 && index > 0) {
        objectTO.connections[0].angleFromPrevious += angleOffset;
    }
    objectTO.numConnections = index;
}

TOs DescriptionConverterService::provideDataTO(
    std::vector<CreatureTO> const& creatureTOs,
    std::vector<GenomeTO> const& genomeTOs,
    std::vector<GeneTO> const& geneTOs,
    std::vector<NodeTO> const& nodeTOs,
    std::vector<ObjectTO> const& objectTOs,
    std::vector<EnergyTO> const& particleTOs,
    std::vector<uint8_t> const& heap) const
{
    TOs result = _collectionTOProvider->provideDataTO(
        {.creatures = creatureTOs.size(),
         .genomes = genomeTOs.size(),
         .genes = geneTOs.size(),
         .nodes = nodeTOs.size(),
         .objects = objectTOs.size(),
         .energyParticles = particleTOs.size(),
         .heap = heap.size()});

    *result.numCreatures = creatureTOs.size();
    *result.numGenomes = genomeTOs.size();
    *result.numGenes = geneTOs.size();
    *result.numNodes = nodeTOs.size();
    *result.numObjects = objectTOs.size();
    *result.numEnergyParticles = particleTOs.size();
    *result.heapSize = heap.size();

    std::memcpy(result.creatures, creatureTOs.data(), creatureTOs.size() * sizeof(CreatureTO));
    std::memcpy(result.genomes, genomeTOs.data(), genomeTOs.size() * sizeof(GenomeTO));
    std::memcpy(result.genes, geneTOs.data(), geneTOs.size() * sizeof(GeneTO));
    std::memcpy(result.nodes, nodeTOs.data(), nodeTOs.size() * sizeof(NodeTO));
    std::memcpy(result.objects, objectTOs.data(), objectTOs.size() * sizeof(ObjectTO));
    std::memcpy(result.energyParticles, particleTOs.data(), particleTOs.size() * sizeof(EnergyTO));
    std::memcpy(result.heap, heap.data(), heap.size());

    return result;
}
