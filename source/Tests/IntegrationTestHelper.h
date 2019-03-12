#pragma once
#include <QEventLoop>

#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/Physics.h"

#include "IntegrationTestFramework.h"

class IntegrationTestHelper
{
public:
	static DataDescription getContent(SimulationAccess* access, IntRect const & rect)
	{
		bool contentReady = false;
		QEventLoop pause;
		auto connection = access->connect(access, &SimulationAccess::dataReadyToRetrieve, [&]() {
			contentReady = true;
			pause.quit();
		});
		ResolveDescription rd;
		rd.resolveCellLinks = true;
		access->requireData(rect, rd);
		if (!contentReady) {
			pause.exec();
		}
		QObject::disconnect(connection);
		return access->retrieveData();
	}

	static void updateData(SimulationAccess* access, DataChangeDescription const& data)
	{
		QEventLoop pause;
		bool finished = false;
		auto connection = access->connect(access, &SimulationAccess::dataUpdated, [&]() {
			finished = true;
			pause.quit();
		});
		access->updateData(data);
		if (!finished) {
			pause.exec();
		}
		QObject::disconnect(connection);
	}

	static void runSimulation(int timesteps, SimulationController* controller)
	{
		QEventLoop pause;
		for (int t = 0; t < timesteps; ++t) {
			bool finished = false;
			auto connection = controller->connect(controller, &SimulationController::nextTimestepCalculated, [&]() {
				finished = true;
				pause.quit();
			});
			controller->calculateSingleTimestep();
			if (!finished) {
				pause.exec();
			}
			QObject::disconnect(connection);

		}
	}

	static unordered_map<uint64_t, ParticleDescription> getParticleByParticleId(DataDescription const& data)
	{
		unordered_map<uint64_t, ParticleDescription> result;
		if (data.particles) {
			std::transform(data.particles->begin(), data.particles->end(), std::inserter(result, result.begin()),
				[](ParticleDescription const& desc) {
					return std::make_pair(desc.id, desc);
				});
		}
		return result;
	}

	static unordered_map<uint64_t, CellDescription> getCellByCellId(DataDescription const& data)
	{
		unordered_map<uint64_t, CellDescription> result;
		if (data.clusters) {
			for (ClusterDescription const& cluster : *data.clusters) {
				for (CellDescription const& cell : *cluster.cells) {
					result.insert_or_assign(cell.id, cell);
				}
			}
		}
		return result;
	}

	static unordered_map<uint64_t, ClusterDescription> getClusterByCellId(DataDescription const& data)
	{
		unordered_map<uint64_t, ClusterDescription> result;
		if (data.clusters) {
			for (ClusterDescription const& cluster : *data.clusters) {
				for (CellDescription const& cell : *cluster.cells) {
					result.insert_or_assign(cell.id, cluster);
				}
			}
		}
		return result;
	}

	static unordered_map<uint64_t, ClusterDescription> getClusterByClusterId(DataDescription const& data)
	{
		unordered_map<uint64_t, ClusterDescription> result;
		if (data.clusters) {
			for (ClusterDescription const& cluster : *data.clusters) {
				result.insert_or_assign(cluster.id, cluster);
			}
		}
		return result;
	}
};