#pragma once
#include <QEventLoop>

#include "Model/Api/SimulationAccess.h"

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
};