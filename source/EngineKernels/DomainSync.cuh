#pragma once

#include <vector>

#include "DomainOps.cuh"
#include "Entities.cuh"
#include "IdMap.cuh"
#include "SensorScans.cuh"

// Records of a sync message. Pointers inside the copied entities are meaningless on the receiving side and are replaced by ids.
struct ObjectRecord
{
    Object object;
    uint64_t connectionIds[MAX_OBJECT_CONNECTIONS];
    uint64_t creatureId;
    int64_t transferPayloadIndex;  // -1 for a ghost copy, otherwise the receiver becomes the owner
    bool applied;                  // Set by the receiver
};

// Heap data of a cell that is only needed by its owner
struct TransferPayload
{
    NeuralNet neuralNet;
    SignalEntry memoryEntries[MAX_CELL_MEMORY_ENTRIES];
};

struct CreatureRecord
{
    Creature creature;
    uint64_t genomeId;
    bool transfer;
};

// A serialized genome: the Genome, its genes, their nodes and the signal entries of memory nodes in one byte block.
// The pointers in it are offsets relative to the start of the block.
struct GenomeEntry
{
    uint64_t genomeId;
    uint64_t byteOffset;
    uint64_t numBytes;
};

struct ParticleRecord
{
    Energy particle;
    bool transfer;  // Otherwise a ghost copy
};

struct SyncMessageCounters
{
    unsigned long long numObjects;
    unsigned long long numTransferPayloads;
    unsigned long long numCreatures;
    unsigned long long numGenomeEntries;
    unsigned long long numGenomeBytes;
    unsigned long long numGenomeRequests;
    unsigned long long numParticles;
    unsigned long long numOps;
    unsigned long long numSensorScanRequests;
    unsigned long long numSensorScanResponses;
    int overflow;
};

struct SyncMessageCapacities
{
    uint64_t objects = 0;
    uint64_t transferPayloads = 0;
    uint64_t creatures = 0;
    uint64_t genomeEntries = 0;
    uint64_t genomeBytes = 0;
    uint64_t genomeRequests = 0;
    uint64_t particles = 0;
    uint64_t ops = 0;
    uint64_t sensorScanRequests = 0;
    uint64_t sensorScanResponses = 0;
    uint64_t bitmapWords = 0;

    bool operator==(SyncMessageCapacities const&) const = default;
};

// The data one domain sends to another domain in a sync round. All sections live in one buffer with fixed offsets.
struct SyncMessage
{
    uint8_t* buffer = nullptr;
    uint64_t bufferSize = 0;
    SyncMessageCapacities capacities;

    SyncMessageCounters* counters = nullptr;
    uint32_t* roiBitmap = nullptr;
    ObjectRecord* objects = nullptr;
    TransferPayload* transferPayloads = nullptr;
    CreatureRecord* creatures = nullptr;
    GenomeEntry* genomeEntries = nullptr;
    uint8_t* genomeBytes = nullptr;
    uint64_t* genomeRequests = nullptr;
    ParticleRecord* particles = nullptr;
    DomainOp* ops = nullptr;
    SensorScanRequest* sensorScanRequests = nullptr;
    SensorScanResponse* sensorScanResponses = nullptr;

    __host__ __inline__ static uint64_t align(uint64_t value) { return (value + 255) / 256 * 256; }

    __host__ __inline__ static uint64_t calcBufferSize(SyncMessageCapacities const& capacities)
    {
        return align(sizeof(SyncMessageCounters)) + align(capacities.bitmapWords * sizeof(uint32_t)) + align(capacities.objects * sizeof(ObjectRecord))
            + align(capacities.transferPayloads * sizeof(TransferPayload)) + align(capacities.creatures * sizeof(CreatureRecord))
            + align(capacities.genomeEntries * sizeof(GenomeEntry)) + align(capacities.genomeBytes) + align(capacities.genomeRequests * sizeof(uint64_t))
            + align(capacities.particles * sizeof(ParticleRecord)) + align(capacities.ops * sizeof(DomainOp))
            + align(capacities.sensorScanRequests * sizeof(SensorScanRequest)) + align(capacities.sensorScanResponses * sizeof(SensorScanResponse));
    }

    // Sets the section pointers for a buffer of calcBufferSize(capacities) bytes
    __host__ __inline__ void assignBuffer(uint8_t* buffer_, SyncMessageCapacities const& capacities_)
    {
        buffer = buffer_;
        capacities = capacities_;
        bufferSize = calcBufferSize(capacities);

        uint64_t offset = 0;
        auto takeSection = [&](uint64_t numBytes) {
            auto result = buffer + offset;
            offset += align(numBytes);
            return result;
        };
        counters = reinterpret_cast<SyncMessageCounters*>(takeSection(sizeof(SyncMessageCounters)));
        roiBitmap = reinterpret_cast<uint32_t*>(takeSection(capacities.bitmapWords * sizeof(uint32_t)));
        objects = reinterpret_cast<ObjectRecord*>(takeSection(capacities.objects * sizeof(ObjectRecord)));
        transferPayloads = reinterpret_cast<TransferPayload*>(takeSection(capacities.transferPayloads * sizeof(TransferPayload)));
        creatures = reinterpret_cast<CreatureRecord*>(takeSection(capacities.creatures * sizeof(CreatureRecord)));
        genomeEntries = reinterpret_cast<GenomeEntry*>(takeSection(capacities.genomeEntries * sizeof(GenomeEntry)));
        genomeBytes = takeSection(capacities.genomeBytes);
        genomeRequests = reinterpret_cast<uint64_t*>(takeSection(capacities.genomeRequests * sizeof(uint64_t)));
        particles = reinterpret_cast<ParticleRecord*>(takeSection(capacities.particles * sizeof(ParticleRecord)));
        ops = reinterpret_cast<DomainOp*>(takeSection(capacities.ops * sizeof(DomainOp)));
        sensorScanRequests = reinterpret_cast<SensorScanRequest*>(takeSection(capacities.sensorScanRequests * sizeof(SensorScanRequest)));
        sensorScanResponses = reinterpret_cast<SensorScanResponse*>(takeSection(capacities.sensorScanResponses * sizeof(SensorScanResponse)));
    }

    // Byte ranges [offset, offset + size) of the used parts of all sections, for copying a message
    struct UsedRange
    {
        uint64_t offset;
        uint64_t size;
    };
    __host__ __inline__ std::vector<UsedRange> getUsedRanges(SyncMessageCounters const& counterValues) const
    {
        auto rangeOf = [&](void const* section, uint64_t size) {
            return UsedRange{static_cast<uint64_t>(reinterpret_cast<uint8_t const*>(section) - buffer), size};
        };
        return {
            rangeOf(counters, sizeof(SyncMessageCounters)),
            rangeOf(roiBitmap, capacities.bitmapWords * sizeof(uint32_t)),
            rangeOf(objects, counterValues.numObjects * sizeof(ObjectRecord)),
            rangeOf(transferPayloads, counterValues.numTransferPayloads * sizeof(TransferPayload)),
            rangeOf(creatures, counterValues.numCreatures * sizeof(CreatureRecord)),
            rangeOf(genomeEntries, counterValues.numGenomeEntries * sizeof(GenomeEntry)),
            rangeOf(genomeBytes, counterValues.numGenomeBytes),
            rangeOf(genomeRequests, counterValues.numGenomeRequests * sizeof(uint64_t)),
            rangeOf(particles, counterValues.numParticles * sizeof(ParticleRecord)),
            rangeOf(ops, counterValues.numOps * sizeof(DomainOp)),
            rangeOf(sensorScanRequests, counterValues.numSensorScanRequests * sizeof(SensorScanRequest)),
            rangeOf(sensorScanResponses, counterValues.numSensorScanResponses * sizeof(SensorScanResponse))};
    }

    // Returns the index of the reserved entries or -1 if the section is full. The counter keeps growing in the latter case,
    // so that the host can derive the required capacity.
    __device__ __inline__ int64_t reserve(unsigned long long* counter, uint64_t capacity, uint64_t numEntries) const
    {
        auto index = atomicAdd(counter, static_cast<unsigned long long>(numEntries));
        if (index + numEntries > capacity) {
            atomicExch(&counters->overflow, 1);
            return -1;
        }
        return static_cast<int64_t>(index);
    }
};

// The values the host needs for its decisions in a sync round, gathered on the device so that one read suffices
struct DomainSyncControl
{
    uint64_t timestep;
    uint64_t numObjects;
    uint64_t numParticles;
    uint64_t heapSize;
    uint64_t objectCapacity;
    uint64_t particleCapacity;
    uint64_t heapCapacity;
    uint64_t numOps;
    uint64_t numSensorContinuations;
    uint64_t numSensorScanRequests;
    uint64_t numPublishedSensorScans;  // Of the pending scans that are published in this round
};

// Synchronization state of one domain that persists across sync rounds
struct DomainSyncData
{
    IdMap<Object> objectMap;
    IdMap<Energy> particleMap;
    IdMap<Creature> creatureMap;
    IdMap<Genome> genomeMap;

    // Device arrays indexed by the other domain
    SyncMessage* outgoingMessages = nullptr;
    SyncMessage* incomingMessages = nullptr;

    // Tiles of the own strip and of the own objects before dilation
    uint32_t* baseBitmap = nullptr;

    // Genome ids to be requested from the given domain in the next round, and genome ids requested by the given domain
    static auto constexpr MaxGenomeRequests = 4096;
    uint64_t* genomeRequestsToSend = nullptr;  // numDomains * MaxGenomeRequests
    int* numGenomeRequestsToSend = nullptr;    // numDomains
    uint64_t* genomesToServe = nullptr;        // numDomains * MaxGenomeRequests
    int* numGenomesToServe = nullptr;          // numDomains

    __device__ __inline__ void queueGenomeRequest(int domain, uint64_t genomeId) const
    {
        auto index = atomicAdd(&numGenomeRequestsToSend[domain], 1);
        if (index < MaxGenomeRequests) {
            genomeRequestsToSend[domain * MaxGenomeRequests + index] = genomeId;
        } else {
            atomicSub(&numGenomeRequestsToSend[domain], 1);
        }
    }

    __device__ __inline__ void queueGenomeToServe(int domain, uint64_t genomeId) const
    {
        auto index = atomicAdd(&numGenomesToServe[domain], 1);
        if (index < MaxGenomeRequests) {
            genomesToServe[domain * MaxGenomeRequests + index] = genomeId;
        } else {
            atomicSub(&numGenomesToServe[domain], 1);
        }
    }
};
