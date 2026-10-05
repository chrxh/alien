#pragma once

#include <functional>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/ArraySizesForGpuEntities.h>
#include <EngineInterface/KernelLaunchSettings.h>

#include <EngineKernels/DomainContext.cuh>
#include <EngineKernels/DomainSync.cuh>

#include "Domain.cuh"

// Capacities of the device arrays that only the sync resizes, kept on the host so that they need not be read back
struct SyncArrayCapacities
{
    uint64_t objectMap = 0;
    uint64_t particleMap = 0;
    uint64_t domainOps = 0;
    uint64_t receivedShockWaves = 0;
    uint64_t sensorContinuations = 0;
    uint64_t sensorScanRequests = 0;
    uint64_t receivedSensorScanRequests = 0;
    uint64_t sensorScanResponses = 0;
    uint64_t pendingSensorScans[2] = {};
};

// Host-side synchronization state of one domain
struct DomainSyncState
{
    DomainSyncData data;
    DomainLayout* layout = nullptr;
    uint32_t* roiBitmaps = nullptr;

    std::vector<SyncMessage> outgoing;    // Index = receiving domain
    std::vector<SyncMessage> incoming;    // Index = sending domain
    std::vector<bool> isIncomingAliased;  // The incoming message is the outgoing buffer of a sender on the same device
    std::vector<SyncMessageCounters> outgoingCounters;

    DomainSyncControl* controlOnDevice = nullptr;
    DomainSyncControl control;  // Read back at the beginning of a sync round
    SyncArrayCapacities capacities;
    ArraySizesForGpuEntities lastIncomingSizes{0, 0, 0};

    double constructorEnergyDemand = 0;  // In the last time step with cell functions
};

// Keeps the domains of a decomposed simulation consistent: hands over objects that leave a domain and refreshes the ghost copies
class DomainSyncService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DomainSyncService);

public:
    using EnsureCapacityFunc = std::function<void(Domain&, ArraySizesForGpuEntities const&)>;

    void init(std::vector<Domain>& domains, int2 const& worldSize, float haloWidth);
    void shutdown(std::vector<Domain>& domains);

    // Every domain holds the whole world after an upload; this keeps the owned objects and the ghosts of each domain
    void distribute(std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings);

    void sync(std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings, EnsureCapacityFunc const& ensureCapacity);

    void updateExternalEnergyShares(std::vector<Domain>& domains, double externalEnergy, bool constructorDemandsMeasured, bool constructorsDrawNext);

private:
    DomainSyncService() = default;

    void readControls(std::vector<Domain>& domains);
    bool reserveMaps(Domain& domain, uint64_t numObjects, uint64_t numParticles);  // Returns true if the maps were resized
    void fillMaps(Domain& domain, KernelLaunchSettings const& launchSettings);
    void calcOwnership(Domain& domain, KernelLaunchSettings const& launchSettings, bool initialAssignment);
    void calcRoi(Domain& domain, KernelLaunchSettings const& launchSettings);
    void pack(Domain& domain, KernelLaunchSettings const& launchSettings);
    void commit(Domain& domain, KernelLaunchSettings const& launchSettings);
    void unpack(Domain& receiver, std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings);
    void finishRound(Domain& domain, std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings);

    bool readOutgoingCountersAndGrowOnOverflow(std::vector<Domain>& domains);
    void transport(std::vector<Domain>& domains);
    void exchangeRoiBitmaps(std::vector<Domain>& domains);
    ArraySizesForGpuEntities calcIncomingSizes(Domain const& receiver, std::vector<Domain> const& domains) const;

    void allocateMessage(SyncMessage& message, SyncMessageCapacities const& capacities);
    void freeMessage(SyncMessage& message);
    void connectIncomingMessages(std::vector<Domain>& domains);
    void uploadMessageDescriptors(Domain& domain);

    uint8_t _round = 0;
    DomainLayout _layout;
};
