#include "Base/ServiceLocator.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/Physics.h"
#include "ModelCpu/ModelCpuBuilderFacade.h"
#include "ModelGpu/ModelGpuBuilderFacade.h"
#include "Gui/DataRepository.h"
#include "Gui/Notifier.h"

#include "Manipulator.h"

namespace
{
	double ForceSensitivity = 0.07;
	double RotationSensitivity = 0.2;
	int CaptureLength = 15;
}

Manipulator::Manipulator(QObject *parent)
	: QObject(parent)
{
}

void Manipulator::init(SimulationContext* context, SimulationAccess* access)
{
	SET_CHILD(_access, access);

	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &Manipulator::dataReadyToRetrieve, Qt::QueuedConnection));
}

void Manipulator::applyForce(QVector2D const& pos, QVector2D const& posDelta)
{
	_mode = Mode::ApplyForce;
	proceedManipulation(pos, posDelta);
}

void Manipulator::applyRotation(QVector2D const & pos, QVector2D const & posDelta)
{
	_mode = Mode::ApplyRotation;
	proceedManipulation(pos, posDelta);
}

void Manipulator::proceedManipulation(QVector2D const& pos, QVector2D const& posDelta)
{
	if (!_waitingForData && posDelta.lengthSquared() > FLOATINGPOINT_MEDIUM_PRECISION) {
		_waitingForData = true;
		_applyAtPos = pos;
		_applyForce = posDelta;
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

namespace
{
	double calcAngularMass(ClusterDescription const& cluster)
	{
		double result = 0.0;
		for (CellDescription const& cell : *cluster.cells) {
			result += (*cell.pos - *cluster.pos).lengthSquared();
		}
		if (cluster.cells->size() == 1) {
			result = 1.0;
		}
		return result;
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
				if (cluster.cells) {
					double mass = cluster.cells->size();
					double angularMass = calcAngularMass(cluster);
					for (CellDescription& cell : *cluster.cells) {
						if ((*cell.pos - _applyAtPos).length() < CaptureLength) {
							QVector2D relPos = *cell.pos - *cluster.pos;
							QVector2D newVel;
							double newAngularVel = 0.0;
							if (_mode == Mode::ApplyForce) {
								CudaPhysics::applyImpulse(_applyForce * ForceSensitivity, relPos, mass, *cluster.vel, angularMass, *cluster.angularVel, newVel, newAngularVel);
								cluster.vel = newVel;
								cluster.angularVel = newAngularVel;
							}
							if (_mode == Mode::ApplyRotation) {
								CudaPhysics::applyImpulse(_applyForce * RotationSensitivity, relPos, mass, *cluster.vel, angularMass, *cluster.angularVel, newVel, newAngularVel);
								cluster.angularVel = newAngularVel;
							}
						}
					}
				}
			}
		}
		if (data.particles) {
			if (_mode == Mode::ApplyForce) {
				for (ParticleDescription& particle : *data.particles) {
					*particle.vel += _applyForce * ForceSensitivity;
				}
			}
		}
		_access->updateData(DataChangeDescription(origData, data));
	}
}
