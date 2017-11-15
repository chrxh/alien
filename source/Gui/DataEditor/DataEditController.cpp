#include <QMatrix4x4>

#include "Model/Api/ChangeDescriptions.h"
#include "Model/Api/SymbolTable.h"
#include "Model/Api/SimulationContext.h"

#include "Gui/DataManipulator.h"
#include "DataEditController.h"
#include "DataEditContext.h"
#include "DataEditModel.h"
#include "DataEditorView.h"

DataEditController::DataEditController(QWidget *parent /*= nullptr*/)
	: QObject(parent)
{
	_view = new DataEditorView(parent);
	_context = new DataEditContext(this);
}

void DataEditController::init(IntVector2D const & upperLeftPosition, DataManipulator * manipulator, SimulationContext* context)
{
	_symbolTable = context->getSymbolTable();
	_model = new DataEditModel(this);
	_model->init(context->getSimulationParameters(), context->getSymbolTable());
	_view->init(upperLeftPosition, _model, this, context->getCellComputerCompiler());
	_manipulator = manipulator;

	connect(_context, &DataEditContext::show, this, &DataEditController::onShow);
	connect(_manipulator, &DataManipulator::notify, this, &DataEditController::notificationFromManipulator);

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

	_manipulator->updateCluster(cluster);
	_manipulator->reconnectSelectedCells();

	uint64_t selectedCellId = _model->getCellToEditRef().id;
	_model->setClusterAndCell(_manipulator->getClusterDescRef(selectedCellId), selectedCellId);

	switchToCellEditor(_manipulator->getCellDescRef(selectedCellId));

	Q_EMIT _manipulator->notify({ DataManipulator::Receiver::Simulation, DataManipulator::Receiver::VisualEditor });
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

	_manipulator->updateCluster(cluster);

	_view->updateDisplay();
	Q_EMIT _manipulator->notify({ DataManipulator::Receiver::Simulation, DataManipulator::Receiver::VisualEditor });
}

void DataEditController::notificationFromParticleTab()
{
	auto& particle = _model->getParticleToEditRef();
	_manipulator->updateParticle(particle);

	Q_EMIT _manipulator->notify({ DataManipulator::Receiver::Simulation, DataManipulator::Receiver::VisualEditor });
}

void DataEditController::notificationFromMetadataTab()
{
	auto& cluster = _model->getClusterToEditRef();
	_manipulator->updateCluster(cluster);

	Q_EMIT _manipulator->notify({ DataManipulator::Receiver::Simulation, DataManipulator::Receiver::VisualEditor });
}

void DataEditController::notificationFromCellComputerTab()
{
	auto& cluster = _model->getClusterToEditRef();
	_manipulator->updateCluster(cluster);

	Q_EMIT _manipulator->notify({ DataManipulator::Receiver::Simulation, DataManipulator::Receiver::VisualEditor });
}

void DataEditController::notificationFromSymbolTab()
{
}

void DataEditController::onShow(bool visible)
{
	_view->show(visible);
}

void DataEditController::notificationFromManipulator(set<DataManipulator::Receiver> const& targets)
{
	if (targets.find(DataManipulator::Receiver::DataEditor) == targets.end()) {
		return;
	}

	auto const& selectedCellIds = _manipulator->getSelectedCellIds();
	auto const& selectedParticleIds = _manipulator->getSelectedParticleIds();
	if (selectedCellIds.size() == 1 && selectedParticleIds.empty()) {

		uint64_t selectedCellId = *selectedCellIds.begin();
		_model->setClusterAndCell(_manipulator->getClusterDescRef(selectedCellId), selectedCellId);
		auto cell = _manipulator->getCellDescRef(selectedCellId);
		switchToCellEditor(_manipulator->getCellDescRef(selectedCellId));
	}
	if (selectedCellIds.empty() && selectedParticleIds.size() == 1) {
		uint64_t selectedParticleId = *selectedParticleIds.begin();
		_model->setParticle(_manipulator->getParticleDescRef(selectedParticleId));
		_view->switchToParticleEditor();
	}
	if (selectedCellIds.size() + selectedParticleIds.size() > 1) {
		_model->setSelectionIds(_manipulator->getSelectedCellIds(), _manipulator->getSelectedParticleIds());
		_view->switchToSelectionEditor();
	}
	if (selectedCellIds.empty() && selectedParticleIds.empty()) {
		_view->switchToNoEditor();
	}
}

void DataEditController::switchToCellEditor(CellDescription const& cell)
{
	bool computerActive = cell.cellFeature->type == Enums::CellFunction::COMPUTER;
	bool tokenActive = cell.tokens && !cell.tokens->empty();
	if (computerActive && tokenActive) {
		_view->switchToCellEditorWithComputerWithToken();
	}
	else if (!computerActive && tokenActive) {
		_view->switchToCellEditorWithoutComputerWithToken();
	}
	else if (computerActive && !tokenActive) {
		_view->switchToCellEditorWithComputerWithoutToken();
	}
	else if (!computerActive && !tokenActive) {
		_view->switchToCellEditorWithoutComputerWithoutToken();
	}
}
