#include "Gui/DataManipulator.h"

#include "DataEditorController.h"
#include "DataEditorContext.h"
#include "DataEditorView.h"

DataEditorController::DataEditorController(QWidget *parent /*= nullptr*/)
	: QObject(parent)
{
	_view = new DataEditorView(parent);
	_context = new DataEditorContext(this);
}

void DataEditorController::init(IntVector2D const & upperLeftPosition, DataManipulator * manipulator)
{
	_view->init(upperLeftPosition);
	_manipulator = manipulator;

	connect(_context, &DataEditorContext::show, this, &DataEditorController::onShow);
	connect(_manipulator, &DataManipulator::notify, this, &DataEditorController::dataUpdatedFromManipulator);

	onShow(false);
}

DataEditorContext * DataEditorController::getContext() const
{
	return _context;
}

void DataEditorController::onShow(bool visible)
{
	_view->show(visible);
}

void DataEditorController::dataUpdatedFromManipulator(set<UpdateTarget> const& targets)
{
	if (targets.find(UpdateTarget::DataEditor) == targets.end()) {
		return;
	}

	auto const& selectedCellIds = _manipulator->getSelectedCellIds();
	auto const& selectedParticleIds = _manipulator->getSelectedParticleIds();
	if (selectedCellIds.size() == 1 && selectedParticleIds.empty()) {
		auto const& clusterDesc = _manipulator->getClusterDescRef(selectedCellIds.front());
		_view->switchToClusterEditor(clusterDesc);
	}
	if (selectedCellIds.empty() && selectedParticleIds.empty()) {
		_view->switchToNoEditor();
	}
}