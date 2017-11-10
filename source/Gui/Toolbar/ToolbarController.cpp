#include "ToolbarController.h"

#include "Model/Api/SimulationContext.h"
#include "Model/Api/SimulationParameters.h"

#include "Gui/DataManipulator.h"

#include "ToolbarView.h"
#include "ToolbarModel.h"
#include "ToolbarContext.h"

ToolbarController::ToolbarController(QWidget* parent)
	: QObject(parent)
{
	_view = new ToolbarView(parent);
	_model = new ToolbarModel(this);
	_context = new ToolbarContext(this);
}

void ToolbarController::init(IntVector2D const & upperLeftPosition, DataManipulator* manipulator, const SimulationContext* context)
{
	_manipulator = manipulator;
	_parameters = context->getSimulationParameters();
	_view->init(upperLeftPosition, this);

	connect(_context, &ToolbarContext::show, this, &ToolbarController::onShow);
	connect(_manipulator, &DataManipulator::notify, this, &ToolbarController::notificationFromManipulator);

	onShow(false);
}

ToolbarContext * ToolbarController::getContext() const
{
	return _context;
}

void ToolbarController::onRequestCell()
{
	_manipulator->addAndSelectCell(_model->getPositionDeltaForNewEntity());
	_manipulator->reconnectSelectedCells();
	Q_EMIT _manipulator->notify({
		DataManipulator::Receiver::DataEditor,
		DataManipulator::Receiver::Simulation,
		DataManipulator::Receiver::VisualEditor,
		DataManipulator::Receiver::Toolbar
	});
}

void ToolbarController::onRequestParticle()
{
	_manipulator->addAndSelectParticle(_model->getPositionDeltaForNewEntity());
	Q_EMIT _manipulator->notify({
		DataManipulator::Receiver::DataEditor,
		DataManipulator::Receiver::Simulation,
		DataManipulator::Receiver::VisualEditor,
		DataManipulator::Receiver::Toolbar
	});
}

void ToolbarController::onDeleteSelection()
{
	_manipulator->deleteSelection();
	Q_EMIT _manipulator->notify({
		DataManipulator::Receiver::DataEditor,
		DataManipulator::Receiver::Simulation,
		DataManipulator::Receiver::VisualEditor,
		DataManipulator::Receiver::Toolbar
	});
}

void ToolbarController::onDeleteExtendedSelection()
{
	_manipulator->deleteExtendedSelection();
	Q_EMIT _manipulator->notify({
		DataManipulator::Receiver::DataEditor,
		DataManipulator::Receiver::Simulation,
		DataManipulator::Receiver::VisualEditor,
		DataManipulator::Receiver::Toolbar
	});
}

void ToolbarController::onRequestToken()
{
	_manipulator->addToken();
	Q_EMIT _manipulator->notify({
		DataManipulator::Receiver::DataEditor,
		DataManipulator::Receiver::Simulation,
		DataManipulator::Receiver::VisualEditor,
		DataManipulator::Receiver::Toolbar
	});
}

void ToolbarController::onShow(bool visible)
{
	_view->setVisible(visible);
}

void ToolbarController::notificationFromManipulator(set<DataManipulator::Receiver> const & targets)
{
	if (targets.find(DataManipulator::Receiver::Toolbar) == targets.end()) {
		return;
	}

	bool isCellSelected = !_manipulator->getSelectedCellIds().empty();
	bool isParticleSelected = !_manipulator->getSelectedParticleIds().empty();
	_view->setEnableDeleteSelections(isCellSelected || isParticleSelected);

	if (_manipulator->getSelectedCellIds().size() == 1) {
		uint64_t selectedCellId = *_manipulator->getSelectedCellIds().begin();
		int numToken = 0;
		if (auto tokens = _manipulator->getCellDescRef(selectedCellId).tokens) {
			numToken = tokens->size();
		}

		_view->setEnableAddToken(numToken < _parameters->cellMaxToken);
		_view->setEnableDeleteToken(numToken > 0);
	}
	else {
		_view->setEnableAddToken(false);
		_view->setEnableDeleteToken(false);
	}
}
