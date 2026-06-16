#include "DescConverterService.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include <cuda_fp16.h>

#include <Base/AlienExceptions.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/NumberGenerator.h>

#include <EngineKernels/TOProvider.cuh>


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
            for (int j = 0; j < NEURONS_PER_CELL; ++j) {
                target[i]._channels[j] = source[i].channels[j];
            }
        }
    }

    // Helper function to copy memory entries from Desc to TO
    template <typename DescEntry, typename TOEntry>
    void copyMemoryEntriesToTO(TOEntry* target, std::vector<DescEntry> const& source)
    {
        for (int i = 0, j = toInt(source.size()); i < j; ++i) {
            auto numChannels = source[i]._channels.size();
            for (int k = 0; k < NEURONS_PER_CELL && k < numChannels; ++k) {
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

    NeuralNetGenomeDesc convert(NeuralNetGenomeTO const& neuralNetworkGenomeTO)
    {
        NeuralNetGenomeDesc result;
        for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
            result._weights[i] = neuralNetworkGenomeTO.weights[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result._biases[i] = neuralNetworkGenomeTO.biases[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result._activationFunctions[i] = neuralNetworkGenomeTO.activationFunctions[i];
        }
        for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
            result._connectionWeights[i] = neuralNetworkGenomeTO.connectionWeights[i];
        }
        return result;
    }

    NeuralNetDesc convert(NeuralNetTO const& neuralNetworkTO)
    {
        NeuralNetDesc result;
        for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
            result._weights[i] = neuralNetworkTO.weights[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result._biases[i] = neuralNetworkTO.biases[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result._activationFunctions[i] = neuralNetworkTO.activationFunctions[i];
        }
        for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
            result._connectionWeights[i] = neuralNetworkTO.connectionWeights[i];
        }
        return result;
    }

    NeuralNetGenomeTO convert(NeuralNetGenomeDesc const& neuralNetworkDesc)
    {
        NeuralNetGenomeTO result;
        for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
            result.weights[i] = neuralNetworkDesc._weights[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result.biases[i] = neuralNetworkDesc._biases[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result.activationFunctions[i] = neuralNetworkDesc._activationFunctions[i];
        }
        for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
            result.connectionWeights[i] = neuralNetworkDesc._connectionWeights[i];
        }
        return result;
    }

    NeuralNetTO convert(NeuralNetDesc const& neuralNetworkDesc)
    {
        CHECK(neuralNetworkDesc._weights.size() == NEURONS_PER_CELL * NEURONS_PER_CELL);
        CHECK(neuralNetworkDesc._biases.size() == NEURONS_PER_CELL);
        CHECK(neuralNetworkDesc._activationFunctions.size() == NEURONS_PER_CELL);
        CHECK(neuralNetworkDesc._connectionWeights.size() == MAX_OBJECT_CONNECTIONS);

        NeuralNetTO result;
        for (int i = 0; i < NEURONS_PER_CELL * NEURONS_PER_CELL; ++i) {
            result.weights[i] = neuralNetworkDesc._weights[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result.biases[i] = neuralNetworkDesc._biases[i];
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            result.activationFunctions[i] = neuralNetworkDesc._activationFunctions[i];
        }
        for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
            result.connectionWeights[i] = neuralNetworkDesc._connectionWeights[i];
        }
        return result;
    }


}

Desc DescConverterService::convertTOtoDescription(TOs const& to) const
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

TOs DescConverterService::convertDescriptionToTO(Desc const& description) const
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

TOs DescConverterService::convertDescriptionToTO(ExtendedObjectDesc const& extendedObject) const
{
    std::vector<ObjectTO> objectTOs;
    std::vector<GenomeTO> genomeTOs;
    std::vector<CreatureTO> creatureTOs;
    std::vector<GeneTO> geneTOs;
    std::vector<NodeTO> nodeTOs;
    std::vector<uint8_t> heap;

    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    std::unordered_map<uint64_t, uint64_t> creatureTOIndexById;

    // Convert genome and creature if available
    if (extendedObject.genome.has_value()) {
        convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, heap, extendedObject.genome.value(), genomeTOIndexById);
    }
    if (extendedObject.creature.has_value()) {
        convertCreatureToTO(creatureTOs, extendedObject.creature.value(), genomeTOIndexById, creatureTOIndexById);
    }

    std::unordered_map<uint64_t, uint64_t> objectIndexTOById;
    convertObjectToTO(objectTOs, heap, objectIndexTOById, extendedObject.object, creatureTOIndexById);

    return provideDataTO(creatureTOs, genomeTOs, geneTOs, nodeTOs, objectTOs, {}, heap);
}

TOs DescConverterService::convertDescriptionToTO(EnergyDesc const& particle) const
{
    std::vector<EnergyTO> particleTOs;
    std::vector<uint8_t> heap;
    addParticle(particleTOs, particle);

    return provideDataTO({}, {}, {}, {}, {}, particleTOs, heap);
}

TOs DescConverterService::convertDescriptionToTO(GenomeDesc const& genome) const
{
    std::vector<GenomeTO> genomeTOs;
    std::vector<GeneTO> geneTOs;
    std::vector<NodeTO> nodeTOs;
    std::vector<uint8_t> heap;

    auto clonedGenome = genome;
    clonedGenome._id = NumberGenerator::get().createEntityId();

    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, heap, clonedGenome, genomeTOIndexById);

    return provideDataTO({}, genomeTOs, geneTOs, nodeTOs, {}, {}, heap);
}

DescConverterService::DescConverterService()
{
    _collectionTOProvider = std::make_shared<_TOProvider>();
}

ObjectDesc DescConverterService::createObjectDesc(TOs const& to, int objectIndex) const
{
    ObjectDesc result(false);

    auto const& objectTO = to.objects[objectIndex];
    result._id = objectTO.id;
    NumberGenerator::get().adaptMaxEntityId(objectTO.id);
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
    result._fixed = objectTO.isFixed();
    result._sticky = objectTO.isSticky();
    result._color = objectTO.color;

    // Handle object type: Solid, Fluid, FreeCell, or Cell
    if (objectTO.type == ObjectType_Solid) {
        SolidDesc solidDesc;
        solidDesc._energy = objectTO.typeData.solid.energy;
        result._type = solidDesc;

    } else if (objectTO.type == ObjectType_Fluid) {
        FluidDesc fluidDesc;
        fluidDesc._energy = objectTO.typeData.fluid.energy;
        fluidDesc._glow = objectTO.typeData.fluid.glow;
        result._type = fluidDesc;

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
        cellDesc._headUpdateId = objectTO.typeData.cell.headUpdateId;
        cellDesc._headCell = objectTO.typeData.cell.headCell;
        cellDesc._event = objectTO.typeData.cell.event;
        cellDesc._eventCounter = objectTO.typeData.cell.eventCounter;
        cellDesc._signalChanges = objectTO.typeData.cell.signalChanges;
        cellDesc._eventPos = {objectTO.typeData.cell.eventPos.x, objectTO.typeData.cell.eventPos.y};
        cellDesc._lastUpdate = objectTO.typeData.cell.lastUpdate;
        cellDesc._concatenationIndex = objectTO.typeData.cell.concatenationIndex;
        cellDesc._branchIndex = objectTO.typeData.cell.branchIndex;

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
        case CellType_Sensor: {
            SensorDesc sensor;
            sensor._autoTrigger = objectTO.typeData.cell.cellTypeData.sensor.autoTrigger;
            sensor._tagForAttackers = objectTO.typeData.cell.cellTypeData.sensor.tagForAttackers;
            sensor._minRange = objectTO.typeData.cell.cellTypeData.sensor.minRange;
            sensor._maxRange = objectTO.typeData.cell.cellTypeData.sensor.maxRange;

            if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_Telemetry) {
                TelemetryDesc telemetry;
                sensor._mode = telemetry;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
                DetectEnergyDesc detectEnergy;
                detectEnergy._minDensity = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectEnergy.minDensity;
                sensor._mode = detectEnergy;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectSolid) {
                DetectSolidDesc detectSolid;
                sensor._mode = detectSolid;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
                DetectFreeCellDesc detectFreeCell;
                detectFreeCell._minDensity = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.minDensity;
                detectFreeCell._restrictToColors = static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectFreeCell.restrictToColors);
                sensor._mode = detectFreeCell;
            } else if (objectTO.typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                DetectCreatureDesc detectCreature;
                detectCreature._minNumCells = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.minNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.minNumCells))
                    : std::nullopt;
                detectCreature._maxNumCells = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells))
                    : std::nullopt;
                detectCreature._restrictToColors = static_cast<int>(objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToColors);
                detectCreature._restrictToLineage = objectTO.typeData.cell.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
                sensor._mode = detectCreature;
            }
            if (objectTO.typeData.cell.cellTypeData.sensor.lastMatchAvailable) {
                SensorLastMatchDesc lastMatchDesc;
                lastMatchDesc._creatureIdPart = objectTO.typeData.cell.cellTypeData.sensor.lastMatch.creatureIdPart;
                lastMatchDesc._pos =
                    RealVector2D{objectTO.typeData.cell.cellTypeData.sensor.lastMatch.pos.x, objectTO.typeData.cell.cellTypeData.sensor.lastMatch.pos.y};
                sensor._lastMatch = lastMatchDesc;
            }

            cellDesc._cellType = sensor;
        } break;
        case CellType_Generator: {
            GeneratorDesc generator;
            generator._additive = objectTO.typeData.cell.cellTypeData.generator.additive;
            generator._minValue = objectTO.typeData.cell.cellTypeData.generator.minValue;
            generator._maxValue = objectTO.typeData.cell.cellTypeData.generator.maxValue;
            generator._timeOffset = objectTO.typeData.cell.cellTypeData.generator.timeOffset;
            if (objectTO.typeData.cell.cellTypeData.generator.mode == GeneratorMode_SquareSignal) {
                SquareSignalDesc squareSignal;
                squareSignal._period = objectTO.typeData.cell.cellTypeData.generator.modeData.squareSignal.period;
                generator._mode = squareSignal;
            } else if (objectTO.typeData.cell.cellTypeData.generator.mode == GeneratorMode_SawtoothSignal) {
                SawtoothSignalDesc sawtoothSignal;
                sawtoothSignal._period = objectTO.typeData.cell.cellTypeData.generator.modeData.sawtoothSignal.period;
                generator._mode = sawtoothSignal;
            }
            generator._numPulses = objectTO.typeData.cell.cellTypeData.generator.numPulses;
            cellDesc._cellType = generator;
        } break;
        case CellType_Attacker: {
            AttackerDesc attacker;
            if (objectTO.typeData.cell.cellTypeData.attacker.mode == AttackerMode_FreeCell) {
                AttackFreeCellDesc attackFreeCell;
                attackFreeCell._restrictToColors = static_cast<int>(objectTO.typeData.cell.cellTypeData.attacker.modeData.attackFreeCell.restrictToColors);
                attacker._mode = attackFreeCell;
            } else if (objectTO.typeData.cell.cellTypeData.attacker.mode == AttackerMode_Creature) {
                AttackCreatureDesc attackCreature;
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
                muscle._mode = bending;
            } else if (objectTO.typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
                ManualBendingDesc bending;
                bending._maxAngleDeviation = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                bending._forwardBackwardRatio = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.forwardBackwardRatio;
                bending._initialAngle = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle != VALUE_NOT_SET_FLOAT
                    ? std::make_optional(objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle)
                    : std::nullopt;
                bending._lastAngleDelta = objectTO.typeData.cell.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
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
            if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_Solid) {
                ReconnectSolidDesc reconnectSolid;
                reconnector._mode = reconnectSolid;
            } else if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
                ReconnectFreeCellDesc reconnectFreeCell;
                reconnectFreeCell._restrictToColors =
                    static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors);
                reconnector._mode = reconnectFreeCell;
            } else if (objectTO.typeData.cell.cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
                ReconnectCreatureDesc reconnectCreature;
                reconnectCreature._minNumCells = objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.minNumCells))
                    : std::nullopt;
                reconnectCreature._maxNumCells = objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells > 0
                    ? std::make_optional(static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells))
                    : std::nullopt;
                reconnectCreature._restrictToColors =
                    static_cast<int>(objectTO.typeData.cell.cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors);
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
                receiver._restrictToColors = static_cast<int>(communicatorTO.modeData.receiver.restrictToColors);
                receiver._restrictToLineage = communicatorTO.modeData.receiver.restrictToLineage;
                communicator._mode = receiver;
            }
            cellDesc._cellType = communicator;
        } break;
        case CellType_Void: {
            cellDesc._cellType = VoidDesc();
        } break;
        }

        // Handle optional constructor field
        if (objectTO.typeData.cell.constructorAvailable) {
            ConstructorDesc constructor;
            constructor._autoTriggerInterval = objectTO.typeData.cell.constructor.autoTriggerInterval > 0
                ? std::make_optional(objectTO.typeData.cell.constructor.autoTriggerInterval)
                : std::nullopt;
            constructor._constructionActivationTime = objectTO.typeData.cell.constructor.constructionActivationTime;
            constructor._constructionAngle = objectTO.typeData.cell.constructor.constructionAngle;
            constructor._provideEnergy = objectTO.typeData.cell.constructor.provideEnergy;
            constructor._reservedEnergy = objectTO.typeData.cell.constructor.reservedEnergy;
            constructor._separation = objectTO.typeData.cell.constructor.separation;
            constructor._numBranches = objectTO.typeData.cell.constructor.numBranches;
            constructor._numConcatenations = objectTO.typeData.cell.constructor.numConcatenations;
            constructor._geneIndex = objectTO.typeData.cell.constructor.geneIndex;
            constructor._lastConstructedCellId = objectTO.typeData.cell.constructor.lastConstructedCellId != VALUE_NOT_SET_UINT64
                ? std::make_optional(objectTO.typeData.cell.constructor.lastConstructedCellId)
                : std::nullopt;
            constructor._currentOffspring = objectTO.typeData.cell.constructor.currentOffspring;
            cellDesc._constructor = constructor;
        }

        auto const& neuralNetworkTO = getFromHeap<NeuralNetTO>(to.heap, objectTO.typeData.cell.neuralNetworkDataIndex);
        cellDesc._neuralNetwork = convert(*neuralNetworkTO);

        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            cellDesc._signal._channels[i] = objectTO.typeData.cell.signal.channels[i];
        }
        cellDesc._signal._numTimesSent = objectTO.typeData.cell.signal.numTimesSent;
        cellDesc._activationTime = objectTO.typeData.cell.activationTime;
        result._type = cellDesc;

    } else {
        CHECK(false);
    }
    return result;
}


NodeDesc DescConverterService::createNodeDesc(TOs const& to, NodeTO const* nodeTO) const
{
    NodeDesc nodeDesc;
    nodeDesc._referenceAngle = nodeTO->referenceAngle;
    nodeDesc._color = nodeTO->color;

    nodeDesc._neuralNetwork = convert(nodeTO->neuralNetwork);

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
    case CellType_Sensor: {
        SensorGenomeDesc sensorDesc;
        sensorDesc._autoTrigger = nodeTO->cellTypeData.sensor.autoTrigger;
        sensorDesc._tagForAttackers = nodeTO->cellTypeData.sensor.tagForAttackers;
        sensorDesc._minRange = nodeTO->cellTypeData.sensor.minRange;
        sensorDesc._maxRange = nodeTO->cellTypeData.sensor.maxRange;

        if (nodeTO->cellTypeData.sensor.mode == SensorMode_Telemetry) {
            TelemetryGenomeDesc telemetry;
            sensorDesc._mode = telemetry;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectEnergy) {
            DetectEnergyGenomeDesc detectEnergy;
            detectEnergy._minDensity = nodeTO->cellTypeData.sensor.modeData.detectEnergy.minDensity;
            sensorDesc._mode = detectEnergy;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectSolid) {
            DetectSolidGenomeDesc detectSolid;
            sensorDesc._mode = detectSolid;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectFreeCell) {
            DetectFreeCellGenomeDesc detectFreeCell;
            detectFreeCell._minDensity = nodeTO->cellTypeData.sensor.modeData.detectFreeCell.minDensity;
            detectFreeCell._restrictToColors = static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectFreeCell.restrictToColors);
            sensorDesc._mode = detectFreeCell;
        } else if (nodeTO->cellTypeData.sensor.mode == SensorMode_DetectCreature) {
            DetectCreatureGenomeDesc detectCreature;
            detectCreature._minNumCells = nodeTO->cellTypeData.sensor.modeData.detectCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectCreature.minNumCells))
                : std::nullopt;
            detectCreature._maxNumCells = nodeTO->cellTypeData.sensor.modeData.detectCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectCreature.maxNumCells))
                : std::nullopt;
            detectCreature._restrictToColors = static_cast<int>(nodeTO->cellTypeData.sensor.modeData.detectCreature.restrictToColors);
            detectCreature._restrictToLineage = nodeTO->cellTypeData.sensor.modeData.detectCreature.restrictToLineage;
            sensorDesc._mode = detectCreature;
        }

        nodeDesc._cellType = sensorDesc;
    } break;
    case CellType_Generator: {
        GeneratorGenomeDesc generatorDesc;
        generatorDesc._additive = nodeTO->cellTypeData.generator.additive;
        generatorDesc._minValue = nodeTO->cellTypeData.generator.minValue;
        generatorDesc._maxValue = nodeTO->cellTypeData.generator.maxValue;
        generatorDesc._timeOffset = nodeTO->cellTypeData.generator.timeOffset;
        if (nodeTO->cellTypeData.generator.mode == GeneratorMode_SquareSignal) {
            SquareSignalGenomeDesc squareSignal;
            squareSignal._period = nodeTO->cellTypeData.generator.modeData.squareSignal.period;
            generatorDesc._mode = squareSignal;
        } else if (nodeTO->cellTypeData.generator.mode == GeneratorMode_SawtoothSignal) {
            SawtoothSignalGenomeDesc sawtoothSignal;
            sawtoothSignal._period = nodeTO->cellTypeData.generator.modeData.sawtoothSignal.period;
            generatorDesc._mode = sawtoothSignal;
        }
        nodeDesc._cellType = generatorDesc;
    } break;
    case CellType_Attacker: {
        AttackerGenomeDesc attackerDesc;
        if (nodeTO->cellTypeData.attacker.mode == AttackerMode_FreeCell) {
            AttackFreeCellGenomeDesc attackFreeCell;
            attackFreeCell._restrictToColors = static_cast<int>(nodeTO->cellTypeData.attacker.modeData.attackFreeCell.restrictToColors);
            attackerDesc._mode = attackFreeCell;
        } else if (nodeTO->cellTypeData.attacker.mode == AttackerMode_Creature) {
            AttackCreatureGenomeDesc attackCreature;
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
        if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_Solid) {
            ReconnectSolidGenomeDesc reconnectSolid;
            reconnectorDesc._mode = reconnectSolid;
        } else if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_FreeCell) {
            ReconnectFreeCellGenomeDesc reconnectFreeCell;
            reconnectFreeCell._restrictToColors = static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectFreeCell.restrictToColors);
            reconnectorDesc._mode = reconnectFreeCell;
        } else if (nodeTO->cellTypeData.reconnector.mode == ReconnectorMode_Creature) {
            ReconnectCreatureGenomeDesc reconnectCreature;
            reconnectCreature._minNumCells = nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.minNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.minNumCells))
                : std::nullopt;
            reconnectCreature._maxNumCells = nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells > 0
                ? std::make_optional(static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.maxNumCells))
                : std::nullopt;
            reconnectCreature._restrictToColors = static_cast<int>(nodeTO->cellTypeData.reconnector.modeData.reconnectCreature.restrictToColors);
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
            receiver._restrictToColors = static_cast<int>(communicatorTO.modeData.receiver.restrictToColors);
            receiver._restrictToLineage = communicatorTO.modeData.receiver.restrictToLineage;
            communicatorDesc._mode = receiver;
        }
        nodeDesc._cellType = communicatorDesc;
    } break;
    case CellType_Void: {
        nodeDesc._cellType = VoidGenomeDesc();
    } break;
    }

    // Handle optional constructor field
    if (nodeTO->constructorAvailable) {
        ConstructorGenomeDesc constructorDesc;
        constructorDesc._autoTriggerInterval =
            nodeTO->constructor.autoTriggerInterval > 0 ? std::make_optional(nodeTO->constructor.autoTriggerInterval) : std::nullopt;
        constructorDesc._geneIndex = nodeTO->constructor.geneIndex;
        constructorDesc._constructionActivationTime = nodeTO->constructor.constructionActivationTime;
        constructorDesc._constructionAngle = nodeTO->constructor.constructionAngle;
        constructorDesc._provideEnergy = nodeTO->constructor.provideEnergy;
        constructorDesc._reservedEnergy = nodeTO->constructor.reservedEnergy;
        constructorDesc._separation = nodeTO->constructor.separation;
        constructorDesc._numBranches = nodeTO->constructor.numBranches;
        constructorDesc._numConcatenations = nodeTO->constructor.numConcatenations;
        nodeDesc._constructor = constructorDesc;
    }

    return nodeDesc;
}

GenomeDesc DescConverterService::createGenomeDesc(TOs const& to, int genomeIndex) const
{
    auto const& genomeTO = to.genomes[genomeIndex];

    GenomeDesc result;
    result._id = genomeTO.id;
    NumberGenerator::get().adaptMaxEntityId(genomeTO.id);
    result._name = char64ToString(genomeTO.name);
    result._lineageId = genomeTO.lineageId;
    NumberGenerator::get().adaptMaxLineageId(genomeTO.lineageId);
    result._prevLineageId = genomeTO.prevLineageId != 0 ? std::make_optional(genomeTO.prevLineageId) : std::nullopt;
    result._frontAngle = genomeTO.frontAngle;
    result._accumulatedMutations = genomeTO.accumulatedMutations;
    for (int i = 0; i < 2; ++i) {
        result._mutationRates._neuronMutations[i]._nodeProbability = genomeTO.mutationRates.neuronMutations[i].nodeProbability;
        result._mutationRates._neuronMutations[i]._weightSigma = genomeTO.mutationRates.neuronMutations[i].weightSigma;
        result._mutationRates._neuronMutations[i]._biasSigma = genomeTO.mutationRates.neuronMutations[i].biasSigma;
        result._mutationRates._neuronMutations[i]._activationFunctionProbability = genomeTO.mutationRates.neuronMutations[i].activationFunctionProbability;
        result._mutationRates._connectionMutations[i]._nodeProbability = genomeTO.mutationRates.connectionMutations[i].nodeProbability;
        result._mutationRates._connectionMutations[i]._sigma = genomeTO.mutationRates.connectionMutations[i].sigma;
        result._mutationRates._cellTypePropertiesMutations[i]._nodeProbability = genomeTO.mutationRates.cellTypePropertiesMutations[i].nodeProbability;
        result._mutationRates._cellTypePropertiesMutations[i]._sigma = genomeTO.mutationRates.cellTypePropertiesMutations[i].sigma;
        result._mutationRates._cellTypePropertiesMutations[i]._discreteChangeProbability =
            genomeTO.mutationRates.cellTypePropertiesMutations[i].discreteChangeProbability;
        result._mutationRates._constructorMutations[i]._nodeProbability = genomeTO.mutationRates.constructorMutations[i].nodeProbability;
        result._mutationRates._constructorMutations[i]._sigma = genomeTO.mutationRates.constructorMutations[i].sigma;
        result._mutationRates._constructorMutations[i]._discreteChangeProbability = genomeTO.mutationRates.constructorMutations[i].discreteChangeProbability;
    }
    result._mutationRates._cellTypeModeMutation._nodeProbability = genomeTO.mutationRates.cellTypeModeMutation.nodeProbability;
    result._mutationRates._cellTypeMutation._nodeProbability = genomeTO.mutationRates.cellTypeMutation.nodeProbability;
    result._mutationRates._voidMutation._nodeProbability = genomeTO.mutationRates.voidMutation.nodeProbability;
    result._mutationRates._appendNodeMutation._geneProbability = genomeTO.mutationRates.appendNodeMutation.geneProbability;
    result._mutationRates._addNodeMutation._geneProbability = genomeTO.mutationRates.addNodeMutation.geneProbability;
    result._mutationRates._trimNodeMutation._geneProbability = genomeTO.mutationRates.trimNodeMutation.geneProbability;
    result._mutationRates._deleteNodeMutation._geneProbability = genomeTO.mutationRates.deleteNodeMutation.geneProbability;
    result._genes.reserve(genomeTO.numGenes);

    CHECK(genomeTO.geneArrayIndex + genomeTO.numGenes <= *to.numGenes);
    for (int j = 0; j < genomeTO.numGenes; ++j) {
        auto geneTO = to.genes + genomeTO.geneArrayIndex + j;

        GeneDesc geneDesc;
        geneDesc._name = char64ToString(geneTO->name);
        geneDesc._shape = geneTO->shape;
        geneDesc._stiffness = geneTO->stiffness;
        geneDesc._connectionDistance = geneTO->connectionDistance;
        geneDesc._homogeneCellType = geneTO->homogeneCellType;

        CHECK(geneTO->nodeArrayIndex + geneTO->numNodes <= *to.numNodes);
        for (int k = 0; k < geneTO->numNodes; ++k) {
            auto nodeTO = to.nodes + geneTO->nodeArrayIndex + k;
            geneDesc._nodes.emplace_back(createNodeDesc(to, nodeTO));
        }

        result._genes.emplace_back(geneDesc);
    }
    return result;
}

CreatureDesc DescConverterService::createCreatureDesc(TOs const& to, int creatureIndex) const
{
    CreatureDesc result;

    auto const& creatureTO = to.creatures[creatureIndex];
    result._id = creatureTO.id;
    NumberGenerator::get().adaptMaxEntityId(creatureTO.id);
    result._ancestorId = creatureTO.ancestorId != VALUE_NOT_SET_UINT64 ? std::make_optional(creatureTO.ancestorId) : std::nullopt;
    result._generation = creatureTO.generation;
    result._numCells = creatureTO.numCells;
    result._mutationState = creatureTO.mutationState;
    result._headUpdateId = creatureTO.headUpdateId;

    return result;
}

EnergyDesc DescConverterService::createEnergyDesc(TOs const& to, int particleIndex) const
{
    auto const& energyParticle = to.energyParticles[particleIndex];
    NumberGenerator::get().adaptMaxEntityId(energyParticle.id);
    return EnergyDesc()
        .id(energyParticle.id)
        .pos({energyParticle.pos.x, energyParticle.pos.y})
        .vel({energyParticle.vel.x, energyParticle.vel.y})
        .energy(energyParticle.energy)
        .color(energyParticle.color);
}

void DescConverterService::convertGenomeToTO(
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
    genomeTO.lineageId = genome._lineageId;
    genomeTO.prevLineageId = genome._prevLineageId.value_or(0);
    genomeTO.frontAngle = genome._frontAngle;
    genomeTO.accumulatedMutations = genome._accumulatedMutations;
    for (int i = 0; i < 2; ++i) {
        genomeTO.mutationRates.neuronMutations[i] = {
            genome._mutationRates._neuronMutations[i]._nodeProbability,
            genome._mutationRates._neuronMutations[i]._weightSigma,
            genome._mutationRates._neuronMutations[i]._biasSigma,
            genome._mutationRates._neuronMutations[i]._activationFunctionProbability};
        genomeTO.mutationRates.connectionMutations[i] = {
            genome._mutationRates._connectionMutations[i]._nodeProbability, genome._mutationRates._connectionMutations[i]._sigma};
        genomeTO.mutationRates.cellTypePropertiesMutations[i] = {
            genome._mutationRates._cellTypePropertiesMutations[i]._nodeProbability,
            genome._mutationRates._cellTypePropertiesMutations[i]._sigma,
            genome._mutationRates._cellTypePropertiesMutations[i]._discreteChangeProbability};
        genomeTO.mutationRates.constructorMutations[i] = {
            genome._mutationRates._constructorMutations[i]._nodeProbability,
            genome._mutationRates._constructorMutations[i]._sigma,
            genome._mutationRates._constructorMutations[i]._discreteChangeProbability};
    }
    genomeTO.mutationRates.cellTypeModeMutation = {genome._mutationRates._cellTypeModeMutation._nodeProbability};
    genomeTO.mutationRates.cellTypeMutation = {genome._mutationRates._cellTypeMutation._nodeProbability};
    genomeTO.mutationRates.voidMutation = {genome._mutationRates._voidMutation._nodeProbability};
    genomeTO.mutationRates.appendNodeMutation = {genome._mutationRates._appendNodeMutation._geneProbability};
    genomeTO.mutationRates.addNodeMutation = {genome._mutationRates._addNodeMutation._geneProbability};
    genomeTO.mutationRates.trimNodeMutation = {genome._mutationRates._trimNodeMutation._geneProbability};
    genomeTO.mutationRates.deleteNodeMutation = {genome._mutationRates._deleteNodeMutation._geneProbability};
    genomeTO.numGenes = toInt(genome._genes.size());
    genomeTO.geneArrayIndex = geneArrayStartIndex;
    genomeTO.genomeIndexOnGpu = VALUE_NOT_SET_UINT64;

    for (auto const& [geneIndex, geneDesc] : genome._genes | boost::adaptors::indexed(0)) {
        GeneTO& geneTO = geneTOs.at(geneArrayStartIndex + geneIndex);

        stringToChar64(geneTO.name, geneDesc._name);
        geneTO.shape = geneDesc._shape;
        geneTO.stiffness = geneDesc._stiffness;
        geneTO.connectionDistance = geneDesc._connectionDistance;
        geneTO.homogeneCellType = geneDesc._homogeneCellType;
        geneTO.numNodes = toInt(geneDesc._nodes.size());

        auto nodeArrayStartIndex = nodeTOs.size();
        nodeTOs.resize(nodeArrayStartIndex + geneDesc._nodes.size());

        geneTO.nodeArrayIndex = nodeArrayStartIndex;
        for (auto const& [nodeIndex, nodeDesc] : geneDesc._nodes | boost::adaptors::indexed(0)) {
            NodeTO& nodeTO = nodeTOs.at(nodeArrayStartIndex + nodeIndex);
            nodeTO.referenceAngle = nodeDesc._referenceAngle;
            nodeTO.color = nodeDesc._color;
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
            case CellType_Sensor: {
                auto const& sensorDesc = std::get<SensorGenomeDesc>(nodeDesc._cellType);
                auto& sensorTO = nodeTO.cellTypeData.sensor;
                sensorTO.autoTrigger = sensorDesc._autoTrigger;
                sensorTO.tagForAttackers = sensorDesc._tagForAttackers;
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
                } else if (sensorTO.mode == SensorMode_DetectSolid) {
                } else if (sensorTO.mode == SensorMode_DetectFreeCell) {
                    auto const& detectFreeCellDesc = std::get<DetectFreeCellGenomeDesc>(sensorDesc._mode);
                    auto& detectFreeCellTO = sensorTO.modeData.detectFreeCell;
                    detectFreeCellTO.minDensity = detectFreeCellDesc._minDensity;
                    detectFreeCellTO.restrictToColors = static_cast<uint16_t>(detectFreeCellDesc._restrictToColors);
                } else if (sensorTO.mode == SensorMode_DetectCreature) {
                    auto const& detectCreatureDesc = std::get<DetectCreatureGenomeDesc>(sensorDesc._mode);
                    auto& detectCreatureTO = sensorTO.modeData.detectCreature;
                    detectCreatureTO.minNumCells = static_cast<uint32_t>(detectCreatureDesc._minNumCells.value_or(0));
                    detectCreatureTO.maxNumCells = static_cast<uint32_t>(detectCreatureDesc._maxNumCells.value_or(0));
                    detectCreatureTO.restrictToColors = static_cast<uint16_t>(detectCreatureDesc._restrictToColors);
                    detectCreatureTO.restrictToLineage = detectCreatureDesc._restrictToLineage;
                }
            } break;
            case CellType_Generator: {
                auto const& generatorDesc = std::get<GeneratorGenomeDesc>(nodeDesc._cellType);
                auto& generatorTO = nodeTO.cellTypeData.generator;
                generatorTO.additive = generatorDesc._additive;
                generatorTO.minValue = generatorDesc._minValue;
                generatorTO.maxValue = generatorDesc._maxValue;
                generatorTO.timeOffset = generatorDesc._timeOffset;
                generatorTO.mode = generatorDesc.getMode();
                if (generatorTO.mode == GeneratorMode_SquareSignal) {
                    auto const& squareSignalDesc = std::get<SquareSignalGenomeDesc>(generatorDesc._mode);
                    generatorTO.modeData.squareSignal.period = squareSignalDesc._period;
                } else if (generatorTO.mode == GeneratorMode_SawtoothSignal) {
                    auto const& sawtoothSignalDesc = std::get<SawtoothSignalGenomeDesc>(generatorDesc._mode);
                    generatorTO.modeData.sawtoothSignal.period = sawtoothSignalDesc._period;
                }
            } break;
            case CellType_Attacker: {
                auto const& attackerDesc = std::get<AttackerGenomeDesc>(nodeDesc._cellType);
                auto& attackerTO = nodeTO.cellTypeData.attacker;
                attackerTO.mode = attackerDesc.getMode();
                if (attackerTO.mode == AttackerMode_FreeCell) {
                    auto const& attackFreeCellDesc = std::get<AttackFreeCellGenomeDesc>(attackerDesc._mode);
                    auto& attackFreeCellTO = attackerTO.modeData.attackFreeCell;
                    attackFreeCellTO.restrictToColors = static_cast<uint16_t>(attackFreeCellDesc._restrictToColors);
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
                if (reconnectorTO.mode == ReconnectorMode_Solid) {
                    // No data to copy
                } else if (reconnectorTO.mode == ReconnectorMode_FreeCell) {
                    auto const& reconnectFreeCellDesc = std::get<ReconnectFreeCellGenomeDesc>(reconnectorDesc._mode);
                    auto& reconnectFreeCellTO = reconnectorTO.modeData.reconnectFreeCell;
                    reconnectFreeCellTO.restrictToColors = static_cast<uint16_t>(reconnectFreeCellDesc._restrictToColors);
                } else if (reconnectorTO.mode == ReconnectorMode_Creature) {
                    auto const& reconnectCreatureDesc = std::get<ReconnectCreatureGenomeDesc>(reconnectorDesc._mode);
                    auto& reconnectCreatureTO = reconnectorTO.modeData.reconnectCreature;
                    reconnectCreatureTO.minNumCells = static_cast<uint32_t>(reconnectCreatureDesc._minNumCells.value_or(0));
                    reconnectCreatureTO.maxNumCells = static_cast<uint32_t>(reconnectCreatureDesc._maxNumCells.value_or(0));
                    reconnectCreatureTO.restrictToColors = static_cast<uint16_t>(reconnectCreatureDesc._restrictToColors);
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
                    communicatorTO.modeData.sender.range = static_cast<uint8_t>(senderDesc._range);
                    communicatorTO.modeData.sender.maxTimesSent = senderDesc._maxTimesSent;
                } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
                    auto const& receiverDesc = std::get<ReceiverGenomeDesc>(communicatorDesc._mode);
                    communicatorTO.modeData.receiver.restrictToColors = static_cast<uint16_t>(receiverDesc._restrictToColors);
                    communicatorTO.modeData.receiver.restrictToLineage = receiverDesc._restrictToLineage;
                }
            } break;
            case CellType_Void: {
            } break;
            }

            // Handle optional constructor field
            nodeTO.constructorAvailable = nodeDesc._constructor.has_value();
            if (nodeDesc._constructor.has_value()) {
                auto const& constructorDesc = nodeDesc._constructor.value();
                nodeTO.constructor.autoTriggerInterval = static_cast<uint32_t>(constructorDesc._autoTriggerInterval.value_or(0));
                nodeTO.constructor.geneIndex = constructorDesc._geneIndex;
                nodeTO.constructor.constructionActivationTime = constructorDesc._constructionActivationTime;
                nodeTO.constructor.constructionAngle = constructorDesc._constructionAngle;
                nodeTO.constructor.provideEnergy = constructorDesc._provideEnergy;
                nodeTO.constructor.reservedEnergy = constructorDesc._reservedEnergy;
                nodeTO.constructor.separation = constructorDesc._separation;
                nodeTO.constructor.numBranches = static_cast<uint8_t>(constructorDesc._numBranches);
                nodeTO.constructor.numConcatenations = constructorDesc._numConcatenations;
            }
        }
    }
}

void DescConverterService::convertCreatureToTO(
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
    creatureTO.headUpdateId = creatureDesc._headUpdateId;
    creatureTO.numCells = creatureDesc._numCells;
    creatureTO.mutationState = creatureDesc._mutationState;
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

void DescConverterService::convertObjectToTO(
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
    objectTO.flags = 0;
    objectTO.setFixed(objectDesc._fixed);
    objectTO.setSticky(objectDesc._sticky);
    objectTO.color = objectDesc._color;

    // Set object type
    objectTO.type = objectDesc.getObjectType();

    // Handle Solid, Fluid, and FreeCell object types
    if (objectTO.type == ObjectType_Solid) {
        SolidDesc const& solidDesc = objectDesc.getSolidRef();
        objectTO.typeData.solid.energy = solidDesc._energy;
    } else if (objectTO.type == ObjectType_Fluid) {
        FluidDesc const& fluidDesc = objectDesc.getFluidRef();
        objectTO.typeData.fluid.energy = fluidDesc._energy;
        objectTO.typeData.fluid.glow = fluidDesc._glow;
    } else if (objectTO.type == ObjectType_FreeCell) {
        FreeCellDesc const& freeCellDesc = objectDesc.getFreeCellRef();
        objectTO.typeData.freeCell.energy = freeCellDesc._energy;
        objectTO.typeData.freeCell.age = freeCellDesc._age;
    } else if (objectTO.type == ObjectType_Cell) {

        // ObjectType_Cell - access cell data
        CellDesc const& cellDesc = objectDesc.getCellRef();

        objectTO.typeData.cell.creatureIndex = creatureTOIndexById.at(cellDesc._creatureId);
        objectTO.typeData.cell.usableEnergy = cellDesc._usableEnergy;
        checkAndCorrectInvalidEnergy(objectTO.typeData.cell.usableEnergy);
        objectTO.typeData.cell.rawEnergy = cellDesc._rawEnergy;
        objectTO.typeData.cell.cellState = cellDesc._cellState;
        objectTO.typeData.cell.cellType = cellDesc.getCellType();
        objectTO.typeData.cell.nodeIndex = cellDesc._nodeIndex;
        objectTO.typeData.cell.parentNodeIndex = cellDesc._parentNodeIndex;
        objectTO.typeData.cell.geneIndex = cellDesc._geneIndex;
        objectTO.typeData.cell.frontAngle = cellDesc._frontAngle.value_or(VALUE_NOT_SET_FLOAT);
        objectTO.typeData.cell.headUpdateId = cellDesc._headUpdateId;
        objectTO.typeData.cell.headCell = cellDesc._headCell;
        objectTO.typeData.cell.age = cellDesc._age;
        objectTO.typeData.cell.activationTime = cellDesc._activationTime;
        objectTO.typeData.cell.event = cellDesc._event;
        objectTO.typeData.cell.eventCounter = cellDesc._eventCounter;
        objectTO.typeData.cell.signalChanges = cellDesc._signalChanges;
        objectTO.typeData.cell.eventPos = {cellDesc._eventPos.x, cellDesc._eventPos.y};
        objectTO.typeData.cell.lastUpdate = cellDesc._lastUpdate;
        objectTO.typeData.cell.concatenationIndex = cellDesc._concatenationIndex;
        objectTO.typeData.cell.branchIndex = cellDesc._branchIndex;

        objectTO.typeData.cell.neuralNetworkDataIndex = heap.size();
        heap.resize(heap.size() + sizeof(NeuralNetTO));
        auto neuralNetworkTO = reinterpret_cast<NeuralNetTO*>(heap.data() + heap.size() - sizeof(NeuralNetTO));
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
        case CellType_Sensor: {
            auto const& sensorDesc = std::get<SensorDesc>(cellDesc._cellType);
            SensorTO& sensorTO = objectTO.typeData.cell.cellTypeData.sensor;
            sensorTO.autoTrigger = sensorDesc._autoTrigger;
            sensorTO.tagForAttackers = sensorDesc._tagForAttackers;
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
            } else if (sensorTO.mode == SensorMode_DetectSolid) {
            } else if (sensorTO.mode == SensorMode_DetectFreeCell) {
                auto const& detectFreeCellDesc = std::get<DetectFreeCellDesc>(sensorDesc._mode);
                DetectFreeCellTO& detectFreeCellTO = sensorTO.modeData.detectFreeCell;
                detectFreeCellTO.minDensity = detectFreeCellDesc._minDensity;
                detectFreeCellTO.restrictToColors = static_cast<uint16_t>(detectFreeCellDesc._restrictToColors);
            } else if (sensorTO.mode == SensorMode_DetectCreature) {
                auto const& detectCreatureDesc = std::get<DetectCreatureDesc>(sensorDesc._mode);
                DetectCreatureTO& detectCreatureTO = sensorTO.modeData.detectCreature;
                detectCreatureTO.minNumCells = static_cast<uint32_t>(detectCreatureDesc._minNumCells.value_or(0));
                detectCreatureTO.maxNumCells = static_cast<uint32_t>(detectCreatureDesc._maxNumCells.value_or(0));
                detectCreatureTO.restrictToColors = static_cast<uint16_t>(detectCreatureDesc._restrictToColors);
                detectCreatureTO.restrictToLineage = detectCreatureDesc._restrictToLineage;
            }
            sensorTO.lastMatchAvailable = sensorDesc._lastMatch.has_value();
            if (sensorDesc._lastMatch.has_value()) {
                sensorTO.lastMatch.creatureIdPart = sensorDesc._lastMatch->_creatureIdPart;
                sensorTO.lastMatch.pos = {sensorDesc._lastMatch->_pos.x, sensorDesc._lastMatch->_pos.y};
            }
        } break;
        case CellType_Generator: {
            auto const& generatorDesc = std::get<GeneratorDesc>(cellDesc._cellType);
            GeneratorTO& generatorTO = objectTO.typeData.cell.cellTypeData.generator;
            generatorTO.additive = generatorDesc._additive;
            generatorTO.minValue = generatorDesc._minValue;
            generatorTO.maxValue = generatorDesc._maxValue;
            generatorTO.timeOffset = generatorDesc._timeOffset;
            generatorTO.mode = generatorDesc.getMode();
            if (generatorTO.mode == GeneratorMode_SquareSignal) {
                auto const& squareSignalDesc = std::get<SquareSignalDesc>(generatorDesc._mode);
                generatorTO.modeData.squareSignal.period = squareSignalDesc._period;
            } else if (generatorTO.mode == GeneratorMode_SawtoothSignal) {
                auto const& sawtoothSignalDesc = std::get<SawtoothSignalDesc>(generatorDesc._mode);
                generatorTO.modeData.sawtoothSignal.period = sawtoothSignalDesc._period;
            }
            generatorTO.numPulses = generatorDesc._numPulses;
        } break;
        case CellType_Attacker: {
            auto const& attackerDesc = std::get<AttackerDesc>(cellDesc._cellType);
            AttackerTO& attackerTO = objectTO.typeData.cell.cellTypeData.attacker;
            attackerTO.mode = attackerDesc.getMode();
            if (attackerTO.mode == AttackerMode_FreeCell) {
                auto const& attackFreeCellDesc = std::get<AttackFreeCellDesc>(attackerDesc._mode);
                attackerTO.modeData.attackFreeCell.restrictToColors = static_cast<uint16_t>(attackFreeCellDesc._restrictToColors);
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
            } else if (muscleTO.mode == MuscleMode_ManualBending) {
                auto const& bendingDesc = std::get<ManualBendingDesc>(muscleDesc._mode);
                ManualBendingTO& bendingTO = muscleTO.modeData.manualBending;
                bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
                bendingTO.forwardBackwardRatio = bendingDesc._forwardBackwardRatio;
                bendingTO.initialAngle = bendingDesc._initialAngle.value_or(VALUE_NOT_SET_FLOAT);
                bendingTO.lastAngleDelta = bendingDesc._lastAngleDelta;
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
            } else if (muscleTO.mode == MuscleMode_ManualCrawling) {
                auto const& crawlingDesc = std::get<ManualCrawlingDesc>(muscleDesc._mode);
                ManualCrawlingTO& crawlingTO = muscleTO.modeData.manualCrawling;
                crawlingTO.maxDistanceDeviation = crawlingDesc._maxDistanceDeviation;
                crawlingTO.forwardBackwardRatio = crawlingDesc._forwardBackwardRatio;
                crawlingTO.initialDistance = crawlingDesc._initialDistance.value_or(VALUE_NOT_SET_FLOAT);
                crawlingTO.lastActualDistance = crawlingDesc._lastActualDistance;
                crawlingTO.lastDistanceDelta = crawlingDesc._lastDistanceDelta;
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
            if (reconnectorTO.mode == ReconnectorMode_Solid) {
                // No data to copy
            } else if (reconnectorTO.mode == ReconnectorMode_FreeCell) {
                auto const& reconnectFreeCellDesc = std::get<ReconnectFreeCellDesc>(reconnectorDesc._mode);
                ReconnectFreeCellTO& reconnectFreeCellTO = reconnectorTO.modeData.reconnectFreeCell;
                reconnectFreeCellTO.restrictToColors = static_cast<uint16_t>(reconnectFreeCellDesc._restrictToColors);
            } else if (reconnectorTO.mode == ReconnectorMode_Creature) {
                auto const& reconnectCreatureDesc = std::get<ReconnectCreatureDesc>(reconnectorDesc._mode);
                ReconnectCreatureTO& reconnectCreatureTO = reconnectorTO.modeData.reconnectCreature;
                reconnectCreatureTO.minNumCells = static_cast<uint32_t>(reconnectCreatureDesc._minNumCells.value_or(0));
                reconnectCreatureTO.maxNumCells = static_cast<uint32_t>(reconnectCreatureDesc._maxNumCells.value_or(0));
                reconnectCreatureTO.restrictToColors = static_cast<uint16_t>(reconnectCreatureDesc._restrictToColors);
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
                communicatorTO.modeData.sender.range = static_cast<uint8_t>(senderDesc._range);
                communicatorTO.modeData.sender.maxTimesSent = senderDesc._maxTimesSent;
            } else if (communicatorTO.mode == CommunicatorMode_Receiver) {
                auto const& receiverDesc = std::get<ReceiverDesc>(communicatorDesc._mode);
                communicatorTO.modeData.receiver.restrictToColors = static_cast<uint16_t>(receiverDesc._restrictToColors);
                communicatorTO.modeData.receiver.restrictToLineage = receiverDesc._restrictToLineage;
            }
        } break;
        case CellType_Void: {
        } break;
        }

        // Handle optional constructor field
        objectTO.typeData.cell.constructorAvailable = cellDesc._constructor.has_value();
        if (cellDesc._constructor.has_value()) {
            auto const& constructorDesc = cellDesc._constructor.value();
            ConstructorTO& constructorTO = objectTO.typeData.cell.constructor;
            constructorTO.autoTriggerInterval = static_cast<uint32_t>(constructorDesc._autoTriggerInterval.value_or(0));
            constructorTO.constructionActivationTime = constructorDesc._constructionActivationTime;
            constructorTO.constructionAngle = constructorDesc._constructionAngle;
            constructorTO.provideEnergy = constructorDesc._provideEnergy;
            constructorTO.reservedEnergy = constructorDesc._reservedEnergy;
            constructorTO.separation = constructorDesc._separation;
            constructorTO.numBranches = static_cast<uint8_t>(constructorDesc._numBranches);
            constructorTO.numConcatenations = constructorDesc._numConcatenations;
            constructorTO.geneIndex = static_cast<uint16_t>(constructorDesc._geneIndex);
            constructorTO.lastConstructedCellId = constructorDesc._lastConstructedCellId.value_or(VALUE_NOT_SET_UINT64);
            constructorTO.currentOffspring = static_cast<uint16_t>(constructorDesc._currentOffspring);
        }

        auto numChannels = cellDesc._signal._channels.size();
        for (int i = 0; i < NEURONS_PER_CELL && i < numChannels; ++i) {
            objectTO.typeData.cell.signal.channels[i] = cellDesc._signal._channels[i];
        }
        objectTO.typeData.cell.signal.numTimesSent = cellDesc._signal._numTimesSent;
    } else {
        CHECK(false);
    }
}

void DescConverterService::addParticle(std::vector<EnergyTO>& particleTOs, EnergyDesc const& particleDesc) const
{
    auto& particleTO = particleTOs.emplace_back();

    particleTO.id = particleDesc._id;
    particleTO.pos = {particleDesc._pos.x, particleDesc._pos.y};
    particleTO.vel = {particleDesc._vel.x, particleDesc._vel.y};
    particleTO.energy = particleDesc._energy;
    checkAndCorrectInvalidEnergy(particleTO.energy);
    particleTO.color = particleDesc._color;
}

void DescConverterService::setConnections(
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

TOs DescConverterService::provideDataTO(
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
