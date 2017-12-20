#include <QAction>

#include "ToolbarController.h"

#include "Model/Api/SimulationContext.h"
#include "Model/Api/SimulationParameters.h"

#include "Gui/DataRepository.h"
#include "Gui/Notifier.h"

#include "ActionHolder.h"
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

void ToolbarController::init(IntVector2D const & upperLeftPosition, Notifier* notifier, DataRepository* manipulator
	, SimulationContext const* context, ActionHolder* actions)
{
	_notifier = notifier;
	_repository = manipulator;
	_parameters = context->getSimulationParameters();
	_actions = actions;
	_view->init(upperLeftPosition, actions, this);

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
	_repository->addAndSelectCell(_model->getPositionDeltaForNewEntity());
	_repository->reconnectSelectedCells();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onRequestParticle()
{
	_repository->addAndSelectParticle(_model->getPositionDeltaForNewEntity());
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onDeleteSelection()
{
	_repository->deleteSelection();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onDeleteExtendedSelection()
{
	_repository->deleteExtendedSelection();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onRequestToken()
{
	_repository->addToken();
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::Toolbar
	}, UpdateDescription::All);
}

void ToolbarController::onDeleteToken()
{
	_repository->deleteToken();
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
	updateActionsEnableState();
}

void ToolbarController::receivedNotifications(set<Receiver> const & targets)
{
	if (targets.find(Receiver::Toolbar) == targets.end()) {
		return;
	}

	int selectedCells = _repository->getSelectedCellIds().size();
	int selectedParticles = _repository->getSelectedParticleIds().size();
	int tokenOfSelectedCell = 0;
	int freeTokenOfSelectedCell = 0;

	if (selectedCells == 1 && selectedParticles == 0) {
		uint64_t selectedCellId = *_repository->getSelectedCellIds().begin();
		if (auto tokens = _repository->getCellDescRef(selectedCellId).tokens) {
			tokenOfSelectedCell = tokens->size();
			freeTokenOfSelectedCell = _parameters->cellMaxToken - tokenOfSelectedCell;
		}
	}

	_model->setEntitySelected(selectedCells == 1 || selectedParticles == 1);
	_model->setCellWithTokenSelected(tokenOfSelectedCell > 0);
	_model->setCellWithFreeTokenSelected(freeTokenOfSelectedCell > 0);
	_model->setCollectionSelected(selectedCells > 0 || selectedParticles > 0);

	updateActionsEnableState();
}

void ToolbarController::updateActionsEnableState()
{
	bool visible = _view->isVisible();
	bool entitySelected = _model->isEntitySelected();
	bool entityCopied = _model->isEntityCopied();
	bool cellWithTokenSelected = _model->isCellWithTokenSelected();
	bool cellWithFreeTokenSelected = _model->isCellWithFreeTokenSelected();
	bool tokenCopied = _model->isTokenCopied();
	bool collectionSelected = _model->isCollectionSelected();
	bool collectionCopied = _model->isCollectionCopied();

	_actions->actionShowCellInfo->setEnabled(visible);

	_actions->actionNewCell->setEnabled(visible);
	_actions->actionNewParticle->setEnabled(visible);
	_actions->actionCopyEntity->setEnabled(visible && entitySelected);
	_actions->actionPasteEntity->setEnabled(visible && entityCopied);
	_actions->actionDeleteEntity->setEnabled(visible && entitySelected);
	_actions->actionNewToken->setEnabled(visible && entitySelected);
	_actions->actionCopyToken->setEnabled(visible && entitySelected);
	_actions->actionPasteToken->setEnabled(visible && cellWithFreeTokenSelected && tokenCopied);
	_actions->actionDeleteToken->setEnabled(visible && cellWithTokenSelected);

	_actions->actionNewRectangle->setEnabled(visible);
	_actions->actionNewHexagon->setEnabled(visible);
	_actions->actionNewParticles->setEnabled(visible);
	_actions->actionLoadCol->setEnabled(visible);
	_actions->actionSaveCol->setEnabled(visible && collectionSelected);
	_actions->actionCopyCol->setEnabled(visible && collectionSelected);
	_actions->actionPasteCol->setEnabled(visible && collectionCopied);
	_actions->actionDeleteCol->setEnabled(visible && collectionSelected);
	_actions->actionDeleteSel->setEnabled(visible && collectionSelected);
	_actions->actionMultiplyRandom->setEnabled(visible && collectionSelected);
	_actions->actionMultiplyArrangement->setEnabled(visible && collectionSelected);
}
