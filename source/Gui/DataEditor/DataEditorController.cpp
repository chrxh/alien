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

void DataEditorController::notificationFromCellEditWidget()
{
}

void DataEditorController::notificationFromClusterEditWidget()
{
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
		_model->editClusterAndCell(_manipulator->getClusterDescRef(selectedCellIds.front()), selectedCellIds.front());
		_view->switchToClusterEditor();
	}
	if (selectedCellIds.size() + selectedParticleIds.size() > 1) {
		_view->switchToNoEditor();
	}
	if (selectedCellIds.empty() && selectedParticleIds.empty()) {
		_view->switchToNoEditor();
	}
}