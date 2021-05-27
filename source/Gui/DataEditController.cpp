#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Base/DebugMacros.h"

#include "EngineInterface/ChangeDescriptions.h"
#include "EngineInterface/SymbolTable.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"

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
    TRY;
	_notifier = notifier;
	_symbolTable = context->getSymbolTable();
	_model = new DataEditModel(this);
	_model->init(manipulator, context->getSimulationParameters(), context->getSymbolTable());
	EngineInterfaceBuilderFacade* basicFacade = ServiceLocator::getInstance().getService<EngineInterfaceBuilderFacade>();
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
    CATCH;
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
    TRY;
    if (auto cluster = _model->getClusterToEditRef()) {

		auto cell = _model->getCellToEditRef();
        if (!cell) {
            return;
		}

        cluster->pos = calcCenterPosOfCells(*cluster);

        _repository->updateCluster(*cluster);
        _repository->reconnectSelectedCells();

        uint64_t selectedCellId = cell->id;
        _model->setClusterAndCell(_repository->getClusterDescRef(selectedCellId), selectedCellId);

        switchToCellEditor(_repository->getCellDescRef(selectedCellId));

        Q_EMIT _notifier->notifyDataRepositoryChanged(
            {Receiver::Simulation, Receiver::VisualEditor}, UpdateDescription::All);
    }
    CATCH;
}

void DataEditController::notificationFromClusterTab()
{
    TRY;

	DataChangeDescription changes = _model->getAndUpdateChanges();
	if (changes.clusters.empty()) {
		return;
	}
	CHECK(changes.clusters.size() == 1);

	auto const& clusterChanges = changes.clusters.front().getValue();
	auto cluster = _model->getClusterToEditRef();
    if (!cluster) {
        return;
    }

	if (clusterChanges.pos) {
		auto delta = clusterChanges.pos.getValue() - clusterChanges.pos.getOldValue();
		for (auto& cell : *cluster->cells) {
			*cell.pos += delta;
		}
	}

	if (clusterChanges.angle) {
		auto delta = clusterChanges.angle.getValue() - clusterChanges.angle.getOldValue();
		QMatrix4x4 transform;
		transform.rotate(delta, 0.0, 0.0, 1.0);
		for (auto& cell : *cluster->cells) {
			auto newRelPos = transform.map(QVector3D(*cell.pos - *cluster->pos)).toVector2D();
			cell.pos = newRelPos + *cluster->pos;
		}
	}

	_repository->updateCluster(*cluster);

	_view->updateDisplay();
	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation, Receiver::VisualEditor }, UpdateDescription::All);

    CATCH;
}

void DataEditController::notificationFromParticleTab()
{
    TRY;
    if (auto particle = _model->getParticleToEditRef()) {
        _repository->updateParticle(*particle);

        Q_EMIT _notifier->notifyDataRepositoryChanged(
            {Receiver::Simulation, Receiver::VisualEditor}, UpdateDescription::All);
    }

    CATCH;
}

void DataEditController::notificationFromMetadataTab()
{
    TRY;
    if (auto cluster = _model->getClusterToEditRef()) {
        _repository->updateCluster(*cluster);

        Q_EMIT _notifier->notifyDataRepositoryChanged(
            {Receiver::Simulation, Receiver::VisualEditor}, UpdateDescription::All);
    }
    CATCH;
}

void DataEditController::notificationFromCellComputerTab()
{
    TRY;
    if (auto cluster = _model->getClusterToEditRef()) {
        _repository->updateCluster(*cluster);

        Q_EMIT _notifier->notifyDataRepositoryChanged(
            {Receiver::Simulation, Receiver::VisualEditor}, UpdateDescription::All);
    }
    CATCH;
}

void DataEditController::notificationFromSymbolTab()
{
    TRY;
    Q_EMIT _notifier->notifyDataRepositoryChanged({Receiver::DataEditor}, UpdateDescription::AllExceptSymbols);
    CATCH;
}

void DataEditController::notificationFromTokenTab()
{
    TRY;
    if (auto cluster = _model->getClusterToEditRef()) {
        _repository->updateCluster(*cluster);

        Q_EMIT _notifier->notifyDataRepositoryChanged({Receiver::Simulation}, UpdateDescription::All);
    }

    CATCH;
}

void DataEditController::onShow(bool visible)
{
    TRY;
    _view->show(visible);
    CATCH;
}

void DataEditController::onRefresh()
{
    TRY;
    _view->updateDisplay();
    CATCH;
}

void DataEditController::receivedExternalNotifications(set<Receiver> const& targets, UpdateDescription update)
{
    TRY;
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
    CATCH;
}

void DataEditController::switchToCellEditor(CellDescription const& cell, UpdateDescription update)
{
    TRY;
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
    CATCH;
}
