#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/NumberGenerator.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/Physics.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpaceProperties.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"

#include "EngineGpu/SimulationControllerGpu.h"
#include "EngineGpu/SimulationAccessGpu.h"
#include "EngineGpu/EngineGpuData.h"
#include "EngineGpu/EngineGpuBuilderFacade.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class IntegrationGpuTestFramework
	: public IntegrationTestFramework
{
public:
    IntegrationGpuTestFramework(IntVector2D const& universeSize = { 900, 600 }, optional<EngineGpuData> const& modelData = boost::none);
	virtual ~IntegrationGpuTestFramework();

protected:
    void check(DataDescription const& origData, DataDescription const& newData) const;

    void checkCellAttributes(DataDescription const& data) const;
    void checkCellConnections(DataDescription const& data) const;
	void checkEnergies(DataDescription const& origData, DataDescription const& newData) const;
    
    Physics::Velocities calcVelocitiesOfClusterPart(ClusterDescription const& cluster, set<uint64_t> const& cellIds) const;
    Physics::Velocities calcVelocitiesOfFusion(ClusterDescription const& cluster1, ClusterDescription const& cluster2) const;
    void setMaxConnections(ClusterDescription& cluster, int maxConnections) const;
    void setCenterPos(ClusterDescription& cluster, QVector2D const& centerPos) const;

    double calcAndCheckEnergy(DataDescription const & data) const;
    double calcAndCheckEnergy(ClusterDescription const& cluster) const;
    double calcKineticEnergy(DataDescription const& data) const;
    double calcKineticEnergy(ClusterDescription const& cluster) const;
    void checkKineticEnergy(DataDescription const& origData, DataDescription const& newData) const;
    void checkEnergyValue(double value) const;

protected:
	double const NearlyZero = FLOATINGPOINT_MEDIUM_PRECISION;

	SimulationControllerGpu* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SpaceProperties* _spaceProp = nullptr;
    SimulationAccessGpu* _access = nullptr;
    DescriptionHelper* _descHelper = nullptr;
};
