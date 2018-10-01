#pragma once
#include <QEventLoop>

#include "ModelInterface/SimulationAccess.h"

class IntegrationTestHelper
{
public:
	static DataDescription getContent(SimulationAccess* access, IntRect const & rect)
	{
		bool contentReady = false;
		QEventLoop pause;
		access->connect(access, &SimulationAccess::dataReadyToRetrieve, [&]() {
			contentReady = true;
			pause.quit();
		});
		ResolveDescription rd;
		rd.resolveCellLinks = true;
		access->requireData(rect, rd);
		if (!contentReady) {
			pause.exec();
		}
		return access->retrieveData();
	}

	static unordered_map<uint64_t, CellDescription> getCellById(DataDescription const& data)
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
};