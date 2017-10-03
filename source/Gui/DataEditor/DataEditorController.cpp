#include <QMatrix4x4>

#include "Model/Api/ChangeDescriptions.h"

#include "Gui/DataManipulator.h"
#include "DataEditorController.h"
#include "DataEditorContext.h"
#include "DataEditorModel.h"
#include "DataEditorView.h"

DataEditorController::DataEditorController(QWidget *parent /*= nullptr*/)
	: QObject(parent)
{
	_view = new DataEditorView(parent);
	_context = new DataEditorContext(this);
}

void DataEditorController::init(IntVector2D const & upperLeftPosition, DataManipulator * manipulator)
{
	_model = new DataEditorModel(this);
	_view->init(upperLeftPosition, _model, this);
	_manipulator = manipulator;

	connect(_context, &DataEditorContext::show, this, &DataEditorController::onShow);
	connect(_manipulator, &DataManipulator::notify, this, &DataEditorController::notificationFromManipulator);

	onShow(false);
}

DataEditorContext * DataEditorController::getContext() const
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

void DataEditorController::notificationFromCellEditWidget()
{
	auto& cluster = _model->getClusterToEditRef();
	cluster.pos = calcCenterPosOfCells(cluster);

	_manipulator->updateCluster(cluster);
	_manipulator->reconnectSelectedCells();

	uint64_t selectedCellId = _model->getCellToEditRef().id;
	_model->editClusterAndCell(_manipulator->getClusterDescRef(selectedCellId), selectedCellId);
	_view->update();
	Q_EMIT _manipulator->notify({ DataManipulator::Receiver::Simulation, DataManipulator::Receiver::VisualEditor });
}

void DataEditorController::notificationFromClusterEditWidget()
{
	DataChangeDescription changes = _model->getAndsUpdateChanges();
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

	_view->update();
	Q_EMIT _manipulator->notify({ DataManipulator::Receiver::Simulation, DataManipulator::Receiver::VisualEditor });
}

void DataEditorController::onShow(bool visible)
{
	_view->show(visible);
}

void DataEditorController::notificationFromManipulator(set<DataManipulator::Receiver> const& targets)
{
	if (targets.find(DataManipulator::Receiver::DataEditor) == targets.end()) {
		return;
	}

	auto const& selectedCellIds = _manipulator->getSelectedCellIds();
	auto const& selectedParticleIds = _manipulator->getSelectedParticleIds();
	if (selectedCellIds.size() == 1 && selectedParticleIds.empty()) {
		uint64_t selectedCellId = selectedCellIds.front();
		_model->editClusterAndCell(_manipulator->getClusterDescRef(selectedCellId), selectedCellId);
		_view->switchToClusterEditor();
	}
	if (selectedCellIds.size() + selectedParticleIds.size() > 1) {
		_view->switchToNoEditor();
	}
	if (selectedCellIds.empty() && selectedParticleIds.empty()) {
		_view->switchToNoEditor();
	}
}