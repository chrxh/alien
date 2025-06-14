#include "DescriptionConverterService.h"

#include <cmath>
#include <algorithm>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include "Base/Exceptions.h"

#include "EngineInterface/NumberGenerator.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineGpuKernels/CollectionTOProvider.cuh"


namespace
{
    template<typename T>
    T* getFromHeap(uint8_t* heap, uint64_t sourceIndex)
    {
        return reinterpret_cast<T*>(&heap[sourceIndex]);
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

    void convert(CollectionTO const& collectionTO, uint64_t sourceSize, uint64_t sourceIndex, std::vector<uint8_t>& target)
    {
        target.resize(sourceSize);
        for (int i = 0; i < sourceSize; ++i) {
            target[i] = collectionTO.heap[sourceIndex + i];
        }
    }

    template<typename Container, typename SizeType>
    void convert(std::vector<uint8_t>& heap, SizeType& targetSize, uint64_t& targetIndex, Container const& source)
    {
        targetSize = source.size();
        if (targetSize > 0) {
            targetIndex = heap.size();
            uint64_t size = source.size();
            for (uint64_t i = 0; i < size; ++i) {
                heap.emplace_back(source.at(i));
            }
        }
    }
}

CollectionDescription DescriptionConverterService::convertTOtoDescription(CollectionTO const& collectionTO) const
{
    CollectionDescription result;

    // Genomes
    std::vector<GenomeDescription_New> genomes;
    std::unordered_map<uint64_t, uint64_t> genomeIdByTOIndex;
    for (int i = 0; i < *collectionTO.numGenomes; ++i) {
        result._genomes.emplace_back(createGenomeDescription(collectionTO, i, genomeIdByTOIndex));
    }

    // Cells
    for (int i = 0; i < *collectionTO.numCells; ++i) {
        result._cells.emplace_back(createCellDescription(collectionTO, i, genomeIdByTOIndex));
    }

    // Particles
    for (int i = 0; i < *collectionTO.numParticles; ++i) {
        ParticleTO const& particle = collectionTO.particles[i];
        result._particles.emplace_back(ParticleDescription()
                                   .id(particle.id)
                                   .pos({particle.pos.x, particle.pos.y})
                                   .vel({particle.vel.x, particle.vel.y})
                                   .energy(particle.energy)
                                   .color(particle.color));
    }

    return result;
}

OverlayDescription DescriptionConverterService::convertTOtoOverlayDescription(CollectionTO const& collectionTO) const
{
    OverlayDescription result;
    result.elements.reserve(*collectionTO.numCells + *collectionTO.numParticles);
    for (int i = 0; i < *collectionTO.numCells; ++i) {
        auto const& cellTO = collectionTO.cells[i];
        OverlayElementDescription element;
        element.id = cellTO.id;
        element.cell = true;
        element.pos = {cellTO.pos.x, cellTO.pos.y};
        element.cellType = static_cast<CellType>(static_cast<unsigned int>(cellTO.cellType) % CellType_Count);
        element.selected = cellTO.selected;
        result.elements.emplace_back(element);
    }

    for (int i = 0; i < *collectionTO.numParticles; ++i) {
        auto const& particleTO = collectionTO.particles[i];
        OverlayElementDescription element;
        element.id = particleTO.id;
        element.cell = false;
        element.pos = {particleTO.pos.x, particleTO.pos.y};
        element.selected = particleTO.selected;
        result.elements.emplace_back(element);
    }
    return result;
}

CollectionTO DescriptionConverterService::convertDescriptionToTO(CollectionDescription const& description) const
{
    std::vector<GenomeTO> genomeTOs;
    std::vector<GeneTO> geneTOs;
    std::vector<NodeTO> nodeTOs;
    std::vector<CellTO> cellTOs;
    std::vector<ParticleTO> particleTOs;
    std::vector<uint8_t> heap;

    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    for (auto const& genome : description._genomes) {
        convertGenomeToTO(genomeTOs, geneTOs, nodeTOs, heap, genome, genomeTOIndexById);
    }

    std::unordered_map<uint64_t, uint64_t> cellIndexTOById;
    for (auto const& cell : description._cells) {
        convertCellToTO(cellTOs, heap, cellIndexTOById, cell, genomeTOIndexById);
    }
    for (auto const& cell : description._cells) {
        if (cell._id != 0) {
            setConnections(cellTOs, cell, cellIndexTOById);
        }
    }
    for (auto const& particle : description._particles) {
        addParticle(particleTOs, particle);
    }

    return provideDataTO(genomeTOs, geneTOs, nodeTOs, cellTOs, particleTOs, heap);
}

CollectionTO DescriptionConverterService::convertDescriptionToTO(CellDescription const& cell) const
{
    std::vector<CellTO> cellTOs;
    std::vector<uint8_t> heap;

    std::unordered_map<uint64_t, uint64_t> cellIndexTOById;
    std::unordered_map<uint64_t, uint64_t> genomeTOIndexById;
    convertCellToTO(cellTOs, heap, cellIndexTOById, cell, genomeTOIndexById);

    return provideDataTO({}, {}, {}, cellTOs, {}, heap);
}

CollectionTO DescriptionConverterService::convertDescriptionToTO(ParticleDescription const& particle) const
{
    std::vector<ParticleTO> particleTOs;
    std::vector<uint8_t> heap;
    addParticle(particleTOs, particle);

    return provideDataTO({}, {}, {}, {}, particleTOs, heap);
}

DescriptionConverterService::DescriptionConverterService()
{
    _collectionTOProvider = std::make_shared<_CollectionTOProvider>();
}

namespace
{
    template <typename T>
    void setInplaceDifference(std::unordered_set<T>& a, std::unordered_set<T> const& b)
    {
        for (auto const& element : b) {
            a.erase(element);
        }
    }
}

CellDescription DescriptionConverterService::createCellDescription(
    CollectionTO const& collectionTO,
    int cellIndex,
    std::unordered_map<uint64_t, uint64_t> const& genomeIdByTOIndex) const
{
    CellDescription result(false);

    auto const& cellTO = collectionTO.cells[cellIndex];
    if (cellTO.hasGenome) {
        result._genomeId = genomeIdByTOIndex.at(cellTO.genomeIndex);
    }
    result._id = cellTO.id;
    result._pos = RealVector2D(cellTO.pos.x, cellTO.pos.y);
    result._vel = RealVector2D(cellTO.vel.x, cellTO.vel.y);
    result._energy = cellTO.energy;
    result._stiffness = cellTO.stiffness;
    std::vector<ConnectionDescription> connections;
    for (int i = 0; i < cellTO.numConnections; ++i) {
        auto const& connectionTO = cellTO.connections[i];
        ConnectionDescription connection;
        if (connectionTO.cellIndex != -1) {
            connection._cellId = collectionTO.cells[connectionTO.cellIndex].id;
        } else {
            connection._cellId = 0;
        }
        connection._distance = connectionTO.distance;
        connection._angleFromPrevious = connectionTO.angleFromPrevious;
        connections.emplace_back(connection);
    }
    result._connections = connections;
    result._livingState = cellTO.livingState;
    result._creatureId = cellTO.creatureId;
    result._mutationId = cellTO.mutationId;
    result._ancestorMutationId = cellTO.ancestorMutationId;
    result._barrier = cellTO.barrier;
    result._sticky = cellTO.sticky;
    result._age = cellTO.age;
    result._color = cellTO.color;
    result._angleToFront = cellTO.angleToFront;
    result._genomeComplexity = cellTO.genomeComplexity;
    result._detectedByCreatureId = cellTO.detectedByCreatureId;
    result._cellTypeUsed = cellTO.cellTypeUsed;
    result._genomeNodeIndex = cellTO.genomeNodeIndex;

    auto const& metacollectionTO = cellTO.metadata;
    auto metadata = CellMetadataDescription();
    if (metacollectionTO.nameSize > 0) {
        auto const name = std::string(reinterpret_cast<char*>(&collectionTO.heap[metacollectionTO.nameDataIndex]), metacollectionTO.nameSize);
        metadata.name(name);
    }
    if (metacollectionTO.descriptionSize > 0) {
        auto const description =
            std::string(reinterpret_cast<char*>(&collectionTO.heap[metacollectionTO.descriptionDataIndex]), metacollectionTO.descriptionSize);
        metadata.description(description);
    }
    result._metadata = metadata;

    switch (cellTO.cellType) {
    case CellType_Structure: {
        StructureCellDescription base;
        result._cellTypeData = base;
    } break;
    case CellType_Free: {
        FreeCellDescription base;
        result._cellTypeData = base;
    } break;
    case CellType_Base: {
        BaseDescription base;
        result._cellTypeData = base;
    } break;
    case CellType_Depot: {
        DepotDescription transmitter;
        transmitter._mode = cellTO.cellTypeData.depot.mode;
        result._cellTypeData = transmitter;
    } break;
    case CellType_Constructor: {
        ConstructorDescription constructor;
        constructor._autoTriggerInterval = cellTO.cellTypeData.constructor.autoTriggerInterval;
        constructor._constructionActivationTime = cellTO.cellTypeData.constructor.constructionActivationTime;
        convert(collectionTO, cellTO.cellTypeData.constructor.genomeSize, cellTO.cellTypeData.constructor.genomeDataIndex, constructor._genome);
        constructor._numInheritedGenomeNodes = cellTO.cellTypeData.constructor.numInheritedGenomeNodes;
        constructor._lastConstructedCellId = cellTO.cellTypeData.constructor.lastConstructedCellId;
        constructor._genomeCurrentNodeIndex = cellTO.cellTypeData.constructor.genomeCurrentNodeIndex;
        constructor._genomeCurrentRepetition = cellTO.cellTypeData.constructor.genomeCurrentRepetition;
        constructor._genomeCurrentBranch = cellTO.cellTypeData.constructor.genomeCurrentBranch;
        constructor._offspringCreatureId = cellTO.cellTypeData.constructor.offspringCreatureId;
        constructor._offspringMutationId = cellTO.cellTypeData.constructor.offspringMutationId;
        constructor._genomeGeneration = cellTO.cellTypeData.constructor.genomeGeneration;
        constructor._constructionAngle1 = cellTO.cellTypeData.constructor.constructionAngle1;
        constructor._constructionAngle2 = cellTO.cellTypeData.constructor.constructionAngle2;
        result._cellTypeData = constructor;
    } break;
    case CellType_Sensor: {
        SensorDescription sensor;
        sensor._autoTriggerInterval = cellTO.cellTypeData.sensor.autoTriggerInterval;
        sensor._minDensity = cellTO.cellTypeData.sensor.minDensity;
        sensor._minRange = cellTO.cellTypeData.sensor.minRange >= 0 ? std::make_optional(cellTO.cellTypeData.sensor.minRange) : std::nullopt;
        sensor._maxRange = cellTO.cellTypeData.sensor.maxRange >= 0 ? std::make_optional(cellTO.cellTypeData.sensor.maxRange) : std::nullopt;
        sensor._restrictToColor =
            cellTO.cellTypeData.sensor.restrictToColor != 255 ? std::make_optional(cellTO.cellTypeData.sensor.restrictToColor) : std::nullopt;
        sensor._restrictToMutants = cellTO.cellTypeData.sensor.restrictToMutants;
        result._cellTypeData = sensor;
    } break;
    case CellType_Oscillator: {
        OscillatorDescription oscillator;
        oscillator._autoTriggerInterval = cellTO.cellTypeData.oscillator.autoTriggerInterval;
        oscillator._alternationInterval = cellTO.cellTypeData.oscillator.alternationInterval;
        oscillator._numPulses = cellTO.cellTypeData.oscillator.numPulses;
        result._cellTypeData = oscillator;
    } break;
    case CellType_Attacker: {
        AttackerDescription attacker;
        result._cellTypeData = attacker;
    } break;
    case CellType_Injector: {
        InjectorDescription injector;
        injector._mode = cellTO.cellTypeData.injector.mode;
        injector._counter = cellTO.cellTypeData.injector.counter;
        convert(collectionTO, cellTO.cellTypeData.injector.genomeSize, cellTO.cellTypeData.injector.genomeDataIndex, injector._genome);
        injector._genomeGeneration = cellTO.cellTypeData.injector.genomeGeneration;
        result._cellTypeData = injector;
    } break;
    case CellType_Muscle: {
        MuscleDescription muscle;
        if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoBending) {
            AutoBendingDescription bending;
            bending._maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
            bending._frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio;
            bending._initialAngle = cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle;
            bending._lastActualAngle = cellTO.cellTypeData.muscle.modeData.autoBending.lastActualAngle;
            bending._forward = cellTO.cellTypeData.muscle.modeData.autoBending.forward;
            bending._activation = cellTO.cellTypeData.muscle.modeData.autoBending.activation;
            bending._activationCountdown = cellTO.cellTypeData.muscle.modeData.autoBending.activationCountdown;
            bending._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
            muscle._mode = bending;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualBending) {
            ManualBendingDescription bending;
            bending._maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
            bending._frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.manualBending.frontBackVelRatio;
            bending._initialAngle = cellTO.cellTypeData.muscle.modeData.manualBending.initialAngle;
            bending._lastActualAngle = cellTO.cellTypeData.muscle.modeData.manualBending.lastActualAngle;
            bending._lastAngleDelta = cellTO.cellTypeData.muscle.modeData.manualBending.lastAngleDelta;
            bending._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.manualBending.impulseAlreadyApplied;
            muscle._mode = bending;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AngleBending) {
            AngleBendingDescription bending;
            bending._maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
            bending._frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.angleBending.frontBackVelRatio;
            bending._initialAngle = cellTO.cellTypeData.muscle.modeData.angleBending.initialAngle;
            muscle._mode = bending;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_AutoCrawling) {
            AutoCrawlingDescription crawling;
            crawling._maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
            crawling._frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio;
            crawling._initialDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.initialDistance;
            crawling._lastActualDistance = cellTO.cellTypeData.muscle.modeData.autoCrawling.lastActualDistance;
            crawling._forward = cellTO.cellTypeData.muscle.modeData.autoCrawling.forward;
            crawling._activation = cellTO.cellTypeData.muscle.modeData.autoCrawling.activation;
            crawling._activationCountdown = cellTO.cellTypeData.muscle.modeData.autoCrawling.activationCountdown;
            crawling._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoCrawling.impulseAlreadyApplied;
            muscle._mode = crawling;
        } else if (cellTO.cellTypeData.muscle.mode == MuscleMode_ManualCrawling) {
            ManualCrawlingDescription crawling;
            crawling._maxDistanceDeviation = cellTO.cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
            crawling._frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio;
            crawling._initialDistance = cellTO.cellTypeData.muscle.modeData.manualCrawling.initialDistance;
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
        result._cellTypeData = muscle;
    } break;
    case CellType_Defender: {
        DefenderDescription defender;
        defender._mode = cellTO.cellTypeData.defender.mode;
        result._cellTypeData = defender;
    } break;
    case CellType_Reconnector: {
        ReconnectorDescription reconnector;
        reconnector._restrictToColor =
            cellTO.cellTypeData.reconnector.restrictToColor != 255 ? std::make_optional(cellTO.cellTypeData.reconnector.restrictToColor) : std::nullopt;
        reconnector._restrictToMutants = cellTO.cellTypeData.reconnector.restrictToMutants;
        result._cellTypeData = reconnector;
    } break;
    case CellType_Detonator: {
        DetonatorDescription detonator;
        detonator._state = cellTO.cellTypeData.detonator.state;
        detonator._countdown = cellTO.cellTypeData.detonator.countdown;
        result._cellTypeData = detonator;
    } break;
    }
    if (cellTO.neuralNetworkDataIndex != CellTO::NeuralNetworkDataIndex_NotSet) {
        auto const& neuralNetworkTO = getFromHeap<NeuralNetworkTO>(collectionTO.heap, cellTO.neuralNetworkDataIndex);
        result._neuralNetwork = convert(*neuralNetworkTO);
    }

    SignalRoutingRestrictionDescription routingRestriction;
    routingRestriction._active = cellTO.signalRoutingRestriction.active;
    routingRestriction._baseAngle = cellTO.signalRoutingRestriction.baseAngle;
    routingRestriction._openingAngle = cellTO.signalRoutingRestriction.openingAngle;
    result._signalRoutingRestriction = routingRestriction;
    result._signalRelaxationTime = cellTO.signalRelaxationTime;
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

GenomeDescription_New DescriptionConverterService::createGenomeDescription(
    CollectionTO const& collectionTO,
    int genomeIndex,
    std::unordered_map<uint64_t, uint64_t>& genomeIdByTOIndex) const
{
    GenomeDescription_New result;

    auto const& genomeTO = collectionTO.genomes[genomeIndex];
    result._id = NumberGenerator::get().createObjectId();
    genomeIdByTOIndex.emplace(genomeIndex, result._id);
    result._genes.reserve(genomeTO.numGenes);
    result._frontAngle = genomeTO.frontAngle;

    for (int i = 0; i < genomeTO.numGenes; ++i) {
        auto geneTO = collectionTO.genes + genomeTO.geneArrayIndex + i;

        GeneDescription geneDesc;
        geneDesc._numBranches = geneTO->numBranches >= 0 ? std::make_optional(geneTO->numBranches) : std::nullopt;
        geneDesc._numBranches = geneTO->numBranches;
        geneDesc._shape = geneTO->shape;
        geneDesc._angleAlignment = geneTO->angleAlignment;
        geneDesc._stiffness = geneTO->stiffness;
        geneDesc._connectionDistance = geneTO->connectionDistance;
        geneDesc._numConcatenations = geneTO->numConcatenations;

        for (int j = 0; j < geneTO->numNodes; ++j) {
            auto nodeTO = collectionTO.nodes + geneTO->nodeArrayIndex + j;

            NodeDescription nodeDesc;
            nodeDesc._referenceAngle = nodeTO->referenceAngle;
            nodeDesc._color = nodeTO->color;
            nodeDesc._numRequiredAdditionalConnections = nodeTO->numRequiredAdditionalConnections;

            auto neuralNetworkGenomeTO = getFromHeap<NeuralNetworkGenomeTO>(collectionTO.heap, nodeTO->neuralNetworkDataIndex);
            nodeDesc._neuralNetwork = convert(*neuralNetworkGenomeTO);
            nodeDesc._numRequiredAdditionalConnections = nodeTO->numRequiredAdditionalConnections;

            switch (nodeTO->cellType) {
            case CellTypeGenome_Base: {
                BaseGenomeDescription baseDesc;
                nodeDesc._cellTypeData = baseDesc;
            } break;
            case CellTypeGenome_Depot: {
                DepotGenomeDescription depotDesc;
                depotDesc._mode = nodeTO->cellTypeData.depot.mode;
                nodeDesc._cellTypeData = depotDesc;
            } break;
            case CellTypeGenome_Constructor: {
                ConstructorGenomeDescription_New constructorDesc;
                constructorDesc._autoTriggerInterval = nodeTO->cellTypeData.constructor.autoTriggerInterval > 0
                    ? std::make_optional(nodeTO->cellTypeData.constructor.autoTriggerInterval)
                    : std::nullopt;
                constructorDesc._constructGeneIndex = nodeTO->cellTypeData.constructor.constructGeneIndex;
                constructorDesc._constructionActivationTime = nodeTO->cellTypeData.constructor.constructionActivationTime;
                nodeDesc._cellTypeData = constructorDesc;
            } break;
            case CellTypeGenome_Sensor: {
                SensorGenomeDescription_New sensorDesc;
                sensorDesc._autoTriggerInterval =
                    nodeTO->cellTypeData.sensor.autoTriggerInterval > 0 ? std::make_optional(nodeTO->cellTypeData.sensor.autoTriggerInterval) : std::nullopt;
                sensorDesc._minDensity = nodeTO->cellTypeData.sensor.minDensity;
                sensorDesc._minRange = nodeTO->cellTypeData.sensor.minRange >= 0 ? std::make_optional(nodeTO->cellTypeData.sensor.minRange) : std::nullopt;
                sensorDesc._maxRange = nodeTO->cellTypeData.sensor.maxRange >= 0 ? std::make_optional(nodeTO->cellTypeData.sensor.maxRange) : std::nullopt;
                sensorDesc._restrictToColor =
                    nodeTO->cellTypeData.sensor.restrictToColor != 255 ? std::make_optional(nodeTO->cellTypeData.sensor.restrictToColor) : std::nullopt;
                sensorDesc._restrictToMutants = nodeTO->cellTypeData.sensor.restrictToMutants;
                nodeDesc._cellTypeData = sensorDesc;
            } break;
            case CellTypeGenome_Oscillator: {
                OscillatorGenomeDescription oscillatorDesc;
                oscillatorDesc._autoTriggerInterval = nodeTO->cellTypeData.oscillator.autoTriggerInterval;
                oscillatorDesc._alternationInterval = nodeTO->cellTypeData.oscillator.alternationInterval;
                nodeDesc._cellTypeData = oscillatorDesc;
            } break;
            case CellTypeGenome_Attacker: {
                AttackerGenomeDescription attackerDesc;
                nodeDesc._cellTypeData = attackerDesc;
            } break;
            case CellTypeGenome_Injector: {
                InjectorGenomeDescription_New injectorDesc;
                injectorDesc._mode = nodeTO->cellTypeData.injector.mode;
                nodeDesc._cellTypeData = injectorDesc;
            } break;
            case CellTypeGenome_Muscle: {
                MuscleGenomeDescription muscleDesc;
                switch (nodeTO->cellTypeData.muscle.mode) {
                case MuscleMode_AutoBending: {
                    AutoBendingGenomeDescription bendingDesc;
                    bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
                    bendingDesc._frontBackVelRatio = nodeTO->cellTypeData.muscle.modeData.autoBending.frontBackVelRatio;
                    muscleDesc._mode = bendingDesc;
                } break;
                case MuscleMode_ManualBending: {
                    ManualBendingGenomeDescription bendingDesc;
                    bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.manualBending.maxAngleDeviation;
                    bendingDesc._frontBackVelRatio = nodeTO->cellTypeData.muscle.modeData.manualBending.frontBackVelRatio;
                    muscleDesc._mode = bendingDesc;
                } break;
                case MuscleMode_AngleBending: {
                    AngleBendingGenomeDescription bendingDesc;
                    bendingDesc._maxAngleDeviation = nodeTO->cellTypeData.muscle.modeData.angleBending.maxAngleDeviation;
                    bendingDesc._frontBackVelRatio = nodeTO->cellTypeData.muscle.modeData.angleBending.frontBackVelRatio;
                    muscleDesc._mode = bendingDesc;
                } break;
                case MuscleMode_AutoCrawling: {
                    AutoCrawlingGenomeDescription crawlingDesc;
                    crawlingDesc._maxDistanceDeviation = nodeTO->cellTypeData.muscle.modeData.autoCrawling.maxDistanceDeviation;
                    crawlingDesc._frontBackVelRatio = nodeTO->cellTypeData.muscle.modeData.autoCrawling.frontBackVelRatio;
                    muscleDesc._mode = crawlingDesc;
                } break;
                case MuscleMode_ManualCrawling: {
                    ManualCrawlingGenomeDescription crawlingDesc;
                    crawlingDesc._maxDistanceDeviation = nodeTO->cellTypeData.muscle.modeData.manualCrawling.maxDistanceDeviation;
                    crawlingDesc._frontBackVelRatio = nodeTO->cellTypeData.muscle.modeData.manualCrawling.frontBackVelRatio;
                    muscleDesc._mode = crawlingDesc;
                } break;
                case MuscleMode_DirectMovement: {
                    DirectMovementGenomeDescription directMovementDesc;
                    muscleDesc._mode = directMovementDesc;
                } break;
                }
                nodeDesc._cellTypeData = muscleDesc;
            } break;
            case CellTypeGenome_Defender: {
                DefenderGenomeDescription defenderDesc;
                defenderDesc._mode = nodeTO->cellTypeData.defender.mode;
                nodeDesc._cellTypeData = defenderDesc;
            } break;
            case CellTypeGenome_Reconnector: {
                ReconnectorGenomeDescription reconnectorDesc;
                reconnectorDesc._restrictToColor = nodeTO->cellTypeData.reconnector.restrictToColor;
                reconnectorDesc._restrictToMutants = nodeTO->cellTypeData.reconnector.restrictToMutants;
                nodeDesc._cellTypeData = reconnectorDesc;
            } break;
            case CellTypeGenome_Detonator: {
                DetonatorGenomeDescription detonatorDesc;
                detonatorDesc._countdown = nodeTO->cellTypeData.detonator.countdown;
                nodeDesc._cellTypeData = detonatorDesc;
            } break;
            }
            geneDesc._nodes.emplace_back(nodeDesc);
        }

        result._genes.emplace_back(geneDesc);
    }

    return result;
}

void DescriptionConverterService::convertGenomeToTO(
    std::vector<GenomeTO>& genomeTOs,
    std::vector<GeneTO>& geneTOs,
    std::vector<NodeTO>& nodeTOs,
    std::vector<uint8_t>& heap,
    GenomeDescription_New const& genomeDesc,
    std::unordered_map<uint64_t, uint64_t>& genomeTOIndexById) const
{
    auto genomeIndex = genomeTOs.size();
    genomeTOs.resize(genomeIndex + 1);

    GenomeTO& genomeTO = genomeTOs.at(genomeIndex);
    genomeTOIndexById.insert_or_assign(genomeDesc._id, genomeIndex);

    auto geneArrayStartIndex = geneTOs.size();
    geneTOs.resize(geneArrayStartIndex + genomeDesc._genes.size());

    genomeTO.numGenes = toInt(genomeDesc._genes.size());
    genomeTO.geneArrayIndex = geneArrayStartIndex;

    for (auto const& [geneIndex, geneDesc] : genomeDesc._genes | boost::adaptors::indexed(0)) {
        GeneTO& geneTO = geneTOs.at(geneArrayStartIndex + geneIndex);

        geneTO.shape = geneDesc._shape;
        geneTO.numBranches = static_cast<int8_t>(geneDesc._numBranches.value_or(-1));
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
            nodeTO.numRequiredAdditionalConnections = nodeDesc._numRequiredAdditionalConnections;

            nodeTO.neuralNetworkDataIndex = heap.size();
            heap.resize(heap.size() + sizeof(NeuralNetworkGenomeTO));
            auto neuralNetworkTO = reinterpret_cast<NeuralNetworkGenomeTO*>(heap.data() + heap.size() - sizeof(NeuralNetworkGenomeTO));
            *neuralNetworkTO = convert(nodeDesc._neuralNetwork);

            nodeTO.cellType = nodeDesc.getCellType();
            switch (nodeDesc.getCellType()) {
            case CellTypeGenome_Base: {
            } break;
            case CellTypeGenome_Depot: {
                auto const& depotDesc = std::get<DepotGenomeDescription>(nodeDesc._cellTypeData);
                auto& depotTO = nodeTO.cellTypeData.depot;
                depotTO.mode = depotDesc._mode;
            } break;
            case CellTypeGenome_Constructor: {
                auto const& constructorDesc = std::get<ConstructorGenomeDescription_New>(nodeDesc._cellTypeData);
                auto& constructorTO = nodeTO.cellTypeData.constructor;
                constructorTO.autoTriggerInterval = static_cast<uint8_t>(constructorDesc._autoTriggerInterval.value_or(0));
                constructorTO.constructGeneIndex = constructorDesc._constructGeneIndex;
                constructorTO.constructionActivationTime = constructorDesc._constructionActivationTime;
                constructorTO.constructionAngle = constructorDesc._constructionAngle;
            } break;
            case CellTypeGenome_Sensor: {
                auto const& sensorDesc = std::get<SensorGenomeDescription_New>(nodeDesc._cellTypeData);
                auto& sensorTO = nodeTO.cellTypeData.sensor;
                sensorTO.autoTriggerInterval = static_cast<uint8_t>(sensorDesc._autoTriggerInterval.value_or(0));
                sensorTO.minDensity = sensorDesc._minDensity;
                sensorTO.minRange = static_cast<int8_t>(sensorDesc._minRange.value_or(-1));
                sensorTO.maxRange = static_cast<int8_t>(sensorDesc._maxRange.value_or(-1));
                sensorTO.restrictToColor = sensorDesc._restrictToColor.value_or(255);
                sensorTO.restrictToMutants = sensorDesc._restrictToMutants;
            } break;
            case CellTypeGenome_Oscillator: {
                auto const& oscillatorDesc = std::get<OscillatorGenomeDescription>(nodeDesc._cellTypeData);
                auto& oscillatorTO = nodeTO.cellTypeData.oscillator;
                oscillatorTO.autoTriggerInterval = oscillatorDesc._autoTriggerInterval;
                oscillatorTO.alternationInterval = oscillatorDesc._alternationInterval;
            } break;
            case CellTypeGenome_Attacker: {
            } break;
            case CellTypeGenome_Injector: {
                auto const& injectorDesc = std::get<InjectorGenomeDescription_New>(nodeDesc._cellTypeData);
                auto& injectorTO = nodeTO.cellTypeData.injector;
                injectorTO.mode = injectorDesc._mode;
            } break;
            case CellTypeGenome_Muscle: {
                auto const& muscleDesc = std::get<MuscleGenomeDescription>(nodeDesc._cellTypeData);
                auto& muscleTO = nodeTO.cellTypeData.muscle;
                muscleTO.mode = muscleDesc.getMode();
                switch (muscleDesc.getMode()) {
                case MuscleMode_AutoBending: {
                    auto const& autoBendingDesc = std::get<AutoBendingGenomeDescription>(muscleDesc._mode);
                    auto& autoBendingTO = muscleTO.modeData.autoBending;
                    autoBendingTO.maxAngleDeviation = autoBendingDesc._maxAngleDeviation;
                    autoBendingTO.frontBackVelRatio = autoBendingDesc._frontBackVelRatio;
                } break;
                case MuscleMode_ManualBending: {
                    auto const& manualBendingDesc = std::get<ManualBendingGenomeDescription>(muscleDesc._mode);
                    auto& manualBendingTO = muscleTO.modeData.manualBending;
                    manualBendingTO.maxAngleDeviation = manualBendingDesc._maxAngleDeviation;
                    manualBendingTO.frontBackVelRatio = manualBendingDesc._frontBackVelRatio;
                } break;
                case MuscleMode_AngleBending: {
                    auto const& angleBendingDesc = std::get<AngleBendingGenomeDescription>(muscleDesc._mode);
                    auto& angleBendingTO = muscleTO.modeData.angleBending;
                    angleBendingTO.maxAngleDeviation = angleBendingDesc._maxAngleDeviation;
                    angleBendingTO.frontBackVelRatio = angleBendingDesc._frontBackVelRatio;
                } break;
                case MuscleMode_AutoCrawling: {
                    auto const& autoCrawlingDesc = std::get<AutoCrawlingGenomeDescription>(muscleDesc._mode);
                    auto& autoCrawlingTO = muscleTO.modeData.autoCrawling;
                    autoCrawlingTO.maxDistanceDeviation = autoCrawlingDesc._maxDistanceDeviation;
                    autoCrawlingTO.frontBackVelRatio = autoCrawlingDesc._frontBackVelRatio;
                } break;
                case MuscleMode_ManualCrawling: {
                    auto const& manualCrawlingDesc = std::get<ManualCrawlingGenomeDescription>(muscleDesc._mode);
                    auto& manualCrawlingTO = muscleTO.modeData.manualCrawling;
                    manualCrawlingTO.maxDistanceDeviation = manualCrawlingDesc._maxDistanceDeviation;
                    manualCrawlingTO.frontBackVelRatio = manualCrawlingDesc._frontBackVelRatio;
                } break;
                case MuscleMode_DirectMovement: {
                } break;
                }
            } break;
            case CellTypeGenome_Defender: {
                auto const& defenderDesc = std::get<DefenderGenomeDescription>(nodeDesc._cellTypeData);
                auto& defenderTO = nodeTO.cellTypeData.defender;
                defenderTO.mode = defenderDesc._mode;
            } break;
            case CellTypeGenome_Reconnector: {
                auto const& reconnectorDesc = std::get<ReconnectorGenomeDescription>(nodeDesc._cellTypeData);
                auto& reconnectorTO = nodeTO.cellTypeData.reconnector;
                reconnectorTO.restrictToColor = reconnectorDesc._restrictToColor.value_or(255);
                reconnectorTO.restrictToMutants = reconnectorDesc._restrictToMutants;
            } break;
            case CellTypeGenome_Detonator: {
                auto const& detonatorDesc = std::get<DetonatorGenomeDescription>(nodeDesc._cellTypeData);
                auto& detonatorTO = nodeTO.cellTypeData.detonator;
                detonatorTO.countdown = detonatorDesc._countdown;
            } break;
            }
        }
    }
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
    std::unordered_map<uint64_t, uint64_t> const& genomeTOIndexById) const
{
    auto cellIndex = cellTOs.size();
    cellTOs.resize(cellIndex + 1);

    CellTO& cellTO = cellTOs.at(cellIndex);
    cellTO.id = cellDesc._id;
    cellTOIndexById.insert_or_assign(cellTO.id, cellIndex);

    cellTO.hasGenome = cellDesc._genomeId.has_value();
    if (cellTO.hasGenome) {
        auto findResult = genomeTOIndexById.find(cellDesc._genomeId.value());
        if (findResult != genomeTOIndexById.end()) {
            cellTO.genomeIndex = findResult->second;
        }
    }
    cellTO.pos = {cellDesc._pos.x, cellDesc._pos.y};
    cellTO.vel = {cellDesc._vel.x, cellDesc._vel.y};
    cellTO.energy = cellDesc._energy;
    checkAndCorrectInvalidEnergy(cellTO.energy);
    cellTO.stiffness = cellDesc._stiffness;
    cellTO.livingState = cellDesc._livingState;
    cellTO.creatureId = cellDesc._creatureId;
    cellTO.mutationId = cellDesc._mutationId;
    cellTO.ancestorMutationId = cellDesc._ancestorMutationId;
    cellTO.cellType = cellDesc.getCellType();
    cellTO.detectedByCreatureId = cellDesc._detectedByCreatureId;
    cellTO.cellTypeUsed = cellDesc._cellTypeUsed;
    cellTO.genomeNodeIndex = cellDesc._genomeNodeIndex;
    cellTO.angleToFront = cellDesc._angleToFront;

    auto cellType = cellDesc.getCellType();
    if (cellDesc._neuralNetwork.has_value()) {
        cellTO.neuralNetworkDataIndex = heap.size();
        heap.resize(heap.size() + sizeof(NeuralNetworkTO));
        auto neuralNetworkTO = reinterpret_cast<NeuralNetworkTO*>(heap.data() + heap.size() - sizeof(NeuralNetworkTO));
        *neuralNetworkTO = convert(*cellDesc._neuralNetwork);
    } else {
        cellTO.neuralNetworkDataIndex = CellTO::NeuralNetworkDataIndex_NotSet;
    }
    switch (cellType) {
    case CellType_Base: {
        BaseTO baseTO;
        cellTO.cellTypeData.base = baseTO;
    } break;
    case CellType_Depot: {
        auto const& transmitterDesc = std::get<DepotDescription>(cellDesc._cellTypeData);
        DepotTO& transmitterTO = cellTO.cellTypeData.depot;
        transmitterTO.mode = transmitterDesc._mode;
    } break;
    case CellType_Constructor: {
        auto const& constructorDesc = std::get<ConstructorDescription>(cellDesc._cellTypeData);
        ConstructorTO& constructorTO = cellTO.cellTypeData.constructor;
        constructorTO.autoTriggerInterval = constructorDesc._autoTriggerInterval;
        constructorTO.constructionActivationTime = constructorDesc._constructionActivationTime;
        CHECK(constructorDesc._genome.size() >= Const::GenomeHeaderSize)
        convert(heap, constructorTO.genomeSize, constructorTO.genomeDataIndex, constructorDesc._genome);
        constructorTO.numInheritedGenomeNodes = static_cast<uint16_t>(constructorDesc._numInheritedGenomeNodes);
        constructorTO.lastConstructedCellId = constructorDesc._lastConstructedCellId;
        constructorTO.genomeCurrentNodeIndex = static_cast<uint16_t>(constructorDesc._genomeCurrentNodeIndex);
        constructorTO.genomeCurrentRepetition = static_cast<uint16_t>(constructorDesc._genomeCurrentRepetition);
        constructorTO.genomeCurrentBranch = static_cast<uint8_t>(constructorDesc._genomeCurrentBranch);
        constructorTO.offspringCreatureId = constructorDesc._offspringCreatureId;
        constructorTO.offspringMutationId = constructorDesc._offspringMutationId;
        constructorTO.genomeGeneration = constructorDesc._genomeGeneration;
        constructorTO.constructionAngle1 = constructorDesc._constructionAngle1;
        constructorTO.constructionAngle2 = constructorDesc._constructionAngle2;
    } break;
    case CellType_Sensor: {
        auto const& sensorDesc = std::get<SensorDescription>(cellDesc._cellTypeData);
        SensorTO& sensorTO = cellTO.cellTypeData.sensor;
        sensorTO.autoTriggerInterval = sensorDesc._autoTriggerInterval;
        sensorTO.restrictToColor = sensorDesc._restrictToColor.value_or(255);
        sensorTO.restrictToMutants = sensorDesc._restrictToMutants;
        sensorTO.minDensity = sensorDesc._minDensity;
        sensorTO.minRange = static_cast<int8_t>(sensorDesc._minRange.value_or(-1));
        sensorTO.maxRange = static_cast<int8_t>(sensorDesc._maxRange.value_or(-1));
    } break;
    case CellType_Oscillator: {
        auto const& oscillatorDesc = std::get<OscillatorDescription>(cellDesc._cellTypeData);
        OscillatorTO& oscillatorTO = cellTO.cellTypeData.oscillator;
        oscillatorTO.autoTriggerInterval = oscillatorDesc._autoTriggerInterval;
        oscillatorTO.alternationInterval = oscillatorDesc._alternationInterval;
        oscillatorTO.numPulses = oscillatorDesc._numPulses;
    } break;
    case CellType_Attacker: {
        auto const& attackerDesc = std::get<AttackerDescription>(cellDesc._cellTypeData);
        AttackerTO& attackerTO = cellTO.cellTypeData.attacker;
    } break;
    case CellType_Injector: {
        auto const& injectorDesc = std::get<InjectorDescription>(cellDesc._cellTypeData);
        InjectorTO& injectorTO = cellTO.cellTypeData.injector;
        injectorTO.mode = injectorDesc._mode;
        injectorTO.counter = injectorDesc._counter;
        CHECK(injectorDesc._genome.size() >= Const::GenomeHeaderSize)
        convert(heap, injectorTO.genomeSize, injectorTO.genomeDataIndex, injectorDesc._genome);
        injectorTO.genomeGeneration = injectorDesc._genomeGeneration;
    } break;
    case CellType_Muscle: {
        auto const& muscleDesc = std::get<MuscleDescription>(cellDesc._cellTypeData);
        MuscleTO& muscleTO = cellTO.cellTypeData.muscle;
        muscleTO.mode = muscleDesc.getMode();
        if (muscleTO.mode == MuscleMode_AutoBending) {
            auto const& bendingDesc = std::get<AutoBendingDescription>(muscleDesc._mode);
            AutoBendingTO& bendingTO = muscleTO.modeData.autoBending;
            bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
            bendingTO.frontBackVelRatio = bendingDesc._frontBackVelRatio;
            bendingTO.initialAngle = bendingDesc._initialAngle;
            bendingTO.lastActualAngle = bendingDesc._lastActualAngle;
            bendingTO.forward = bendingDesc._forward;
            bendingTO.activation = bendingDesc._activation;
            bendingTO.activationCountdown = bendingDesc._activationCountdown;
            bendingTO.impulseAlreadyApplied = bendingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_ManualBending) {
            auto const& bendingDesc = std::get<ManualBendingDescription>(muscleDesc._mode);
            ManualBendingTO& bendingTO = muscleTO.modeData.manualBending;
            bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
            bendingTO.frontBackVelRatio = bendingDesc._frontBackVelRatio;
            bendingTO.initialAngle = bendingDesc._initialAngle;
            bendingTO.lastActualAngle = bendingDesc._lastActualAngle;
            bendingTO.lastAngleDelta = bendingDesc._lastAngleDelta;
            bendingTO.impulseAlreadyApplied = bendingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_AngleBending) {
            auto const& bendingDesc = std::get<AngleBendingDescription>(muscleDesc._mode);
            AngleBendingTO& bendingTO = muscleTO.modeData.angleBending;
            bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
            bendingTO.frontBackVelRatio = bendingDesc._frontBackVelRatio;
            bendingTO.initialAngle = bendingDesc._initialAngle;
        } else if (muscleTO.mode == MuscleMode_AutoCrawling) {
            auto const& crawlingDesc = std::get<AutoCrawlingDescription>(muscleDesc._mode);
            AutoCrawlingTO& crawlingTO = muscleTO.modeData.autoCrawling;
            crawlingTO.maxDistanceDeviation = crawlingDesc._maxDistanceDeviation;
            crawlingTO.frontBackVelRatio = crawlingDesc._frontBackVelRatio;
            crawlingTO.initialDistance = crawlingDesc._initialDistance;
            crawlingTO.lastActualDistance = crawlingDesc._lastActualDistance;
            crawlingTO.forward = crawlingDesc._forward;
            crawlingTO.activation = crawlingDesc._activation;
            crawlingTO.activationCountdown = crawlingDesc._activationCountdown;
            crawlingTO.impulseAlreadyApplied = crawlingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_ManualCrawling) {
            auto const& crawlingDesc = std::get<ManualCrawlingDescription>(muscleDesc._mode);
            ManualCrawlingTO& crawlingTO = muscleTO.modeData.manualCrawling;
            crawlingTO.maxDistanceDeviation = crawlingDesc._maxDistanceDeviation;
            crawlingTO.frontBackVelRatio = crawlingDesc._frontBackVelRatio;
            crawlingTO.initialDistance = crawlingDesc._initialDistance;
            crawlingTO.lastActualDistance = crawlingDesc._lastActualDistance;
            crawlingTO.lastDistanceDelta = crawlingDesc._lastDistanceDelta;
            crawlingTO.impulseAlreadyApplied = crawlingDesc._impulseAlreadyApplied;
        } else if (muscleTO.mode == MuscleMode_DirectMovement) {
        }
        muscleTO.lastMovementX = muscleDesc._lastMovementX;
        muscleTO.lastMovementY = muscleDesc._lastMovementY;
    } break;
    case CellType_Defender: {
        auto const& defenderDesc = std::get<DefenderDescription>(cellDesc._cellTypeData);
        DefenderTO& defenderTO = cellTO.cellTypeData.defender;
        defenderTO.mode = defenderDesc._mode;
    } break;
    case CellType_Reconnector: {
        auto const& reconnectorDesc = std::get<ReconnectorDescription>(cellDesc._cellTypeData);
        ReconnectorTO& reconnectorTO = cellTO.cellTypeData.reconnector;
        reconnectorTO.restrictToColor = toUInt8(reconnectorDesc._restrictToColor.value_or(255));
        reconnectorTO.restrictToMutants = reconnectorDesc._restrictToMutants;
    } break;
    case CellType_Detonator: {
        auto const& detonatorDesc = std::get<DetonatorDescription>(cellDesc._cellTypeData);
        DetonatorTO& detonatorTO = cellTO.cellTypeData.detonator;
        detonatorTO.state = detonatorDesc._state;
        detonatorTO.countdown = detonatorDesc._countdown;
    } break;
    }
    cellTO.signalRoutingRestriction.active = cellDesc._signalRoutingRestriction._active;
    cellTO.signalRoutingRestriction.baseAngle = cellDesc._signalRoutingRestriction._baseAngle;
    cellTO.signalRoutingRestriction.openingAngle = cellDesc._signalRoutingRestriction._openingAngle;
    cellTO.signalRelaxationTime = cellDesc._signalRelaxationTime;
    cellTO.signal.active = cellDesc._signal.has_value();
    if (cellTO.signal.active) {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cellDesc._signal->_channels[i];
        }
    }
    cellTO.activationTime = cellDesc._activationTime;
    cellTO.numConnections = 0;
    cellTO.barrier = cellDesc._barrier;
    cellTO.sticky = cellDesc._sticky;
    cellTO.age = cellDesc._age;
    cellTO.color = cellDesc._color;
    cellTO.genomeComplexity = cellDesc._genomeComplexity;
    convert(heap, cellTO.metadata.nameSize, cellTO.metadata.nameDataIndex, cellDesc._metadata._name);
    convert(heap, cellTO.metadata.descriptionSize, cellTO.metadata.descriptionDataIndex, cellDesc._metadata._description);
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
        if (connection._cellId != 0) {
            cellTO.connections[index].cellIndex = cellIndexByIds.at(connection._cellId);
            cellTO.connections[index].distance = connection._distance;
            cellTO.connections[index].angleFromPrevious = connection._angleFromPrevious + angleOffset;
            ++index;
            angleOffset = 0;
        } else {
            angleOffset += connection._angleFromPrevious;
        }
    }
    if (angleOffset != 0 && index > 0) {
        cellTO.connections[0].angleFromPrevious += angleOffset;
    }
    cellTO.numConnections = index;
}

CollectionTO DescriptionConverterService::provideDataTO(
    std::vector<GenomeTO> const& genomeTOs,
    std::vector<GeneTO> const& geneTOs,
    std::vector<NodeTO> const& nodeTOs,
    std::vector<CellTO> const& cellTOs,
    std::vector<ParticleTO> const& particleTOs,
    std::vector<uint8_t> const& heap) const
{
    CollectionTO result = _collectionTOProvider->provideDataTO(
        {.genomes = genomeTOs.size(),
         .genes = geneTOs.size(),
         .nodes = nodeTOs.size(),
         .cells = cellTOs.size(),
         .particles = particleTOs.size(),
         .heap = heap.size()});

    *result.numGenomes = genomeTOs.size();
    *result.numGenes = geneTOs.size();
    *result.numNodes = nodeTOs.size();
    *result.numCells = cellTOs.size();
    *result.numParticles = particleTOs.size();
    *result.heapSize = heap.size();

    std::memcpy(result.genomes, genomeTOs.data(), genomeTOs.size() * sizeof(GenomeTO));
    std::memcpy(result.genes, geneTOs.data(), geneTOs.size() * sizeof(GeneTO));
    std::memcpy(result.nodes, nodeTOs.data(), nodeTOs.size() * sizeof(NodeTO));
    std::memcpy(result.cells, cellTOs.data(), cellTOs.size() * sizeof(CellTO));
    std::memcpy(result.particles, particleTOs.data(), particleTOs.size() * sizeof(ParticleTO));
    std::memcpy(result.heap, heap.data(), heap.size());

    return result;
}
