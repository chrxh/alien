#include "ToolbarController.h"

#include "Model/Api/SimulationContext.h"
#include "Model/Api/SimulationParameters.h"

#include "Gui/DataController.h"
#include "Gui/Notifier.h"

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

void ToolbarController::init(IntVector2D const & upperLeftPosition, Notifier* notifier, DataController* manipulator, const SimulationContext* context)
{
	_notifier = notifier;
	_manipulator = manipulator;
	_parameters = context->getSimulationParameters();
	_view->init(upperLeftPosition, this);

	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(_context, &ToolbarContext::show, this, &ToolbarController::onShow));
	_connections.push_back(connect(_notifier, &Notifier::notify, this, &ToolbarController::receivedNotifications));

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
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onRequestParticle()
{
	_manipulator->addAndSelectParticle(_model->getPositionDeltaForNewEntity());
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onDeleteSelection()
{
	_manipulator->deleteSelection();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onDeleteExtendedSelection()
{
	_manipulator->deleteExtendedSelection();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onRequestToken()
{
	_manipulator->addToken();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onDeleteToken()
{
	_manipulator->deleteToken();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onToggleCellInfo(bool showInfo)
{
	Q_EMIT _notifier->toggleCellInfo(showInfo);
}

void ToolbarController::onShow(bool visible)
{
	_view->setVisible(visible);
}

void ToolbarController::receivedNotifications(set<Receiver> const & targets)
{
	if (targets.find(Receiver::Toolbar) == targets.end()) {
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
