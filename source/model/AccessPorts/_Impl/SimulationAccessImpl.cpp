#include "global/ServiceLocator.h"
#include "model/AccessPorts/Descriptions.h"
#include "model/AccessPorts/LightDescriptions.h"
#include "model/AccessPorts/AccessPortFactory.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitContext.h"
#include "model/context/UnitThreadController.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/entities/CellCluster.h"

#include "SimulationAccessImpl.h"

template class SimulationAccessImpl<DataDescription>;
template class SimulationAccessImpl<DataLightDescription>;

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContext*>(context);
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::addData(DataDescriptionType const& desc)
{
	AccessPortFactory* portFactory = ServiceLocator::getInstance().getService<AccessPortFactory>();

	_context->getUnitThreadController()->lock();
	auto grid = _context->getUnitGrid();
	for (auto const& clusterdesc : desc.clusters) {
		auto unitContext = grid->getUnitOfMapPos(clusterdesc.pos)->getContext();
		auto cluster = portFactory->buildFromDescription(clusterdesc, unitContext);
		cluster->drawCellsToMap();
		unitContext->getClustersRef().push_back(cluster);
	}
	for (auto const& particleDesc : desc.particles) {
		auto unitContext = grid->getUnitOfMapPos(particleDesc.pos)->getContext();
		auto particle = portFactory->buildFromDescription(particleDesc, unitContext);
		unitContext->getEnergyParticlesRef().push_back(particle);
	}
	_context->getUnitThreadController()->unlock();
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::removeData(DataDescriptionType const & desc)
{
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::updateData(DataDescriptionType const & desc)
{
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::getData(IntRect rect, DataDescriptionType & result)
{
	auto grid = _context->getUnitGrid();
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(QVector3D(rect.p1.x, rect.p1.y, 0));
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(QVector3D(rect.p2.x, rect.p2.y, 0));
	_context->getUnitThreadController()->lock();
	_context->getUnitThreadController()->unlock();
}
