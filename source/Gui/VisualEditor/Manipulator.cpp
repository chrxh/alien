#include "Base/ServiceLocator.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationAccess.h"

#include "Gui/DataRepository.h"
#include "Gui/Notifier.h"

#include "Manipulator.h"

namespace
{
	double Sensitivity = 0.05;
	int CaptureLength = 10;
}

Manipulator::Manipulator(QObject *parent)
	: QObject(parent)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();

	_access = facade->buildSimulationAccess();
}

void Manipulator::init(SimulationContext* context)
{
	_access->init(context);

	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &Manipulator::dataReadyToRetrieve));
}

void Manipulator::applyForce(QVector2D const& pos, QVector2D const& force)
{
	if (!_waitingForData && force.lengthSquared() > FLOATINGPOINT_MEDIUM_PRECISION) {
		_waitingForData = true;
		_applyAtPos = pos;
		_applyForce = force;
		IntVector2D intPos(pos);
		IntRect updateRect({
			{ intPos.x - CaptureLength / 2, intPos.y - CaptureLength / 2 },
			{ intPos.x + CaptureLength / 2, intPos.y + CaptureLength / 2 }
		});

		ResolveDescription resolveDesc;
		resolveDesc.resolveCellLinks = false;
		_access->requireData(updateRect, resolveDesc);
	}
}

void Manipulator::dataReadyToRetrieve()
{
	if (_waitingForData) {
		_waitingForData = false;
		DataDescription const& origData = _access->retrieveData();
		DataDescription data = origData;
		if (data.clusters) {
			for (ClusterDescription& cluster : *data.clusters) {
				*cluster.vel += _applyForce * Sensitivity;
			}
		}
		_access->updateData(DataChangeDescription(origData, data));
	}
}
