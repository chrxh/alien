#include <QMatrix4x4>

#include "Base/ServiceLocator.h"

#include "ModelBasic/ChangeDescriptions.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"

#include "Gui/DataRepository.h"
#include "Gui/Notifier.h"

#include "DataEditController.h"
#include "DataEditContext.h"
#include "DataEditModel.h"
#include "DataEditView.h"

DataEditController::DataEditController(QWidget *parent /*= nullptr*/)
	: QObject(parent)
{
	_view = new DataEditView(parent);
	_context = new DataEditContext(this);
}

void DataEditController::init(IntVector2D const & upperLeftPosition, Notifier* notifier, DataRepository * manipulator, SimulationContext* context)
{
	_notifier = notifier;
	_symbolTable = context->getSymbolTable();
	_model = new DataEditModel(this);
	_model->init(manipulator, context->getSimulationParameters(), context->getSymbolTable());
	ModelBasicBuilderFacade* basicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
	CellComputerCompiler* compiler = basicFacade->buildCellComputerCompiler(context->getSymbolTable(), context->getSimulationParameters());
	_view->init(upperLeftPosition, _model, this, compiler);
	_repository = manipulator;

	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(_context, &DataEditContext::show, this, &DataEditController::onShow));
	_connections.push_back(connect(_context, &DataEditContext::refresh, this, &DataEditController::onRefresh));
	_connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &DataEditController::receivedExternalNotifications));

	onShow(false);
}

DataEditContext * DataEditController::getContext() const
{
	return _context;
}

namespace
{
	QVector2D calcCenterPosOfCells(ClusterDescription const& cluster)
	{
		QVector2D result;
		for (auto const& cell : *cluster.cells) {
			result += *cell.pos;
		}
		result  /= cluster.cells->size();
		return result;
	}
}

void DataEditController::notificationFromCellTab()
{
	auto& cluster = _model->getClusterToEditRef();
	cluster.pos = calcCenterPosOfCells(cluster);

	_repository->updateCluster(cluster);
	_repository->reconnectSelectedCells();

	uint64_t selectedCellId = _model->getCellToEditRef().id;
	_model->setClusterAndCell(_repository->getClusterDescRef(selectedCellId), selectedCellId);

	switchToCellEditor(_repository->getCellDescRef(selectedCellId));

	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation, Receiver::VisualEditor }, UpdateDescription::All);
}

void DataEditController::notificationFromClusterTab()
{
	DataChangeDescription changes = _model->getAndUpdateChanges();
	if (changes.clusters.empty()) {
		return;
	}
	CHECK(changes.clusters.size() == 1);

	auto const& clusterChanges = changes.clusters.front().getValue();
	auto& cluster = _model->getClusterToEditRef();

	if (clusterChanges.pos) {
		auto delta = clusterChanges.pos.getValue() - clusterChanges.pos.getOldValue();
		for (auto& cell : *cluster.cells) {
			*cell.pos += delta;
		}
	}

	if (clusterChanges.angle) {
		auto delta = clusterChanges.angle.getValue() - clusterChanges.angle.getOldValue();
		QMatrix4x4 transform;
		transform.rotate(delta, 0.0, 0.0, 1.0);
		for (auto& cell : *cluster.cells) {
			auto newRelPos = transform.map(QVector3D(*cell.pos - *cluster.pos)).toVector2D();
			cell.pos = newRelPos + *cluster.pos;
		}
	}

	_repository->updateCluster(cluster);

	_view->updateDisplay();
	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation, Receiver::VisualEditor }, UpdateDescription::All);
}

void DataEditController::notificationFromParticleTab()
{
	auto& particle = _model->getParticleToEditRef();
	_repository->updateParticle(particle);

	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation, Receiver::VisualEditor }, UpdateDescription::All);
}

void DataEditController::notificationFromMetadataTab()
{
	auto& cluster = _model->getClusterToEditRef();
	_repository->updateCluster(cluster);

	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation, Receiver::VisualEditor }, UpdateDescription::All);
}

void DataEditController::notificationFromCellComputerTab()
{
	auto& cluster = _model->getClusterToEditRef();
	_repository->updateCluster(cluster);

	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation, Receiver::VisualEditor }, UpdateDescription::All);
}

void DataEditController::notificationFromSymbolTab()
{
	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::DataEditor }, UpdateDescription::AllExceptSymbols);
}

void DataEditController::notificationFromTokenTab()
{
	auto& cluster = _model->getClusterToEditRef();
	_repository->updateCluster(cluster);

	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation }, UpdateDescription::All);
}

void DataEditController::onShow(bool visible)
{
	_view->show(visible);
}

void DataEditController::onRefresh()
{
	_view->updateDisplay();
}

void DataEditController::receivedExternalNotifications(set<Receiver> const& targets, UpdateDescription update)
{
	if (targets.find(Receiver::DataEditor) == targets.end()) {
		return;
	}

	auto const& selectedCellIds = _repository->getSelectedCellIds();
	auto const& selectedParticleIds = _repository->getSelectedParticleIds();
	if (selectedCellIds.size() == 1 && selectedParticleIds.empty()) {

		uint64_t selectedCellId = *selectedCellIds.begin();
		_model->setClusterAndCell(_repository->getClusterDescRef(selectedCellId), selectedCellId);
		auto cell = _repository->getCellDescRef(selectedCellId);
		switchToCellEditor(_repository->getCellDescRef(selectedCellId), update);
	}
	if (selectedCellIds.empty() && selectedParticleIds.size() == 1) {
		uint64_t selectedParticleId = *selectedParticleIds.begin();
		_model->setParticle(_repository->getParticleDescRef(selectedParticleId));
		_view->switchToParticleEditor();
	}
	if (selectedCellIds.size() + selectedParticleIds.size() > 1) {
		_model->setSelectionIds(_repository->getSelectedCellIds(), _repository->getSelectedParticleIds());
		_view->switchToSelectionEditor();
	}
	if (selectedCellIds.empty() && selectedParticleIds.empty()) {
		_view->switchToNoEditor();
	}
}

void DataEditController::switchToCellEditor(CellDescription const& cell, UpdateDescription update)
{
	auto const computerActive = (Enums::CellFunction::COMPUTER == cell.cellFeature->getType());
	bool tokenActive = cell.tokens && !cell.tokens->empty();
	if (computerActive && tokenActive) {
		_view->switchToCellEditorWithComputerWithToken(update);
	}
	else if (!computerActive && tokenActive) {
		_view->switchToCellEditorWithoutComputerWithToken(update);
	}
	else if (computerActive && !tokenActive) {
		_view->switchToCellEditorWithComputerWithoutToken();
	}
	else if (!computerActive && !tokenActive) {
		_view->switchToCellEditorWithoutComputerWithoutToken();
	}
}
