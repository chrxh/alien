#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/NumberGenerator.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/DescriptionHelper.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationContext.h"

#include "ModelGpu/SimulationControllerGpu.h"
#include "ModelGpu/SimulationAccessGpu.h"
#include "ModelGpu/ModelGpuData.h"
#include "ModelGpu/ModelGpuBuilderFacade.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class SimulationGpuTest
	: public IntegrationTestFramework
{
public:
	SimulationGpuTest();
	virtual ~SimulationGpuTest();

protected:
	void checkEnergy(DataDescription const& origData, DataDescription const& newData) const;
	void checkDistancesToConnectingCells(DataDescription const& data) const;
	void checkKineticEnergy(DataDescription const& origData, DataDescription const& newData) const;
	Physics::Velocities calcVelocitiesOfClusterPart(ClusterDescription const& cluster, set<uint64_t> const& cellIds) const;
	Physics::Velocities calcVelocitiesOfFusion(ClusterDescription const& cluster1, ClusterDescription const& cluster2) const;
	double calcEnergy(DataDescription const& data) const;
	double calcEnergy(ClusterDescription const& cluster) const;
	double calcKineticEnergy(DataDescription const& data) const;
	double calcKineticEnergy(ClusterDescription const& cluster) const;
	void setMaxConnections(ClusterDescription& cluster, int maxConnections) const;

protected:
	double const NearlyZero = FLOATINGPOINT_MEDIUM_PRECISION;

	SimulationControllerGpu* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SpaceProperties* _spaceProp = nullptr;
	SimulationAccessGpu* _access = nullptr;
};
