#include <QFileDialog>
#include <QMessageBox>
#include <QAction>
#include <QInputDialog>
#include <QClipboard>
#include <QTextStream>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SymbolTable.h"
#include "EngineInterface/Physics.h"
#include "EngineInterface/SerializationHelper.h"
#include "EngineInterface/DescriptionFactory.h"

#include "Web/WebAccess.h"

#include "ToolbarController.h"
#include "ToolbarContext.h"
#include "SimulationViewWidget.h"
#include "DataEditController.h"
#include "DataEditContext.h"
#include "NewSimulationDialog.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"
#include "ComputationSettingsDialog.h"
#include "NewRectangleDialog.h"
#include "NewHexagonDialog.h"
#include "NewParticlesDialog.h"
#include "RandomMultiplierDialog.h"
#include "GridMultiplierDialog.h"
#include "MonitorController.h"
#include "Settings.h"
#include "InfoController.h"
#include "MainController.h"
#include "MainModel.h"
#include "MainView.h"
#include "Notifier.h"
#include "ActionModel.h"
#include "ActionController.h"
#include "ActionHolder.h"
#include "SimulationConfig.h"
#include "WebSimulationController.h"

ActionController::ActionController(QObject * parent)
	: QObject(parent)
{
	_model = new ActionModel(this);
}

void ActionController::init(
    MainController* mainController,
    MainModel* mainModel,
    MainView* mainView,
    SimulationViewWidget* visualEditor,
    Serializer* serializer,
    InfoController* infoController,
    DataEditController* dataEditor,
    ToolbarController* toolbar,
    MonitorController* monitor,
    DataRepository* repository,
    Notifier* notifier,
    WebSimulationController* webSimController)
{
    auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
    auto numberGenerator = factory->buildRandomNumberGenerator();
	numberGenerator->init();

	_mainController = mainController;
	_mainModel = mainModel;
	_mainView = mainView;
	_simulationViewWidget = visualEditor;
	_serializer = serializer;
	_infoController = infoController;
	_dataEditor = dataEditor;
	_toolbar = toolbar;
	_monitor = monitor;
	_repository = repository;
	_notifier = notifier;
    _webSimController = webSimController;
	SET_CHILD(_numberGenerator, numberGenerator);

	connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &ActionController::receivedNotifications);

	auto actions = _model->getActionHolder();
	connect(actions->actionNewSimulation, &QAction::triggered, this, &ActionController::onNewSimulation);
    connect(actions->actionWebSimulation, &QAction::toggled, this, &ActionController::onWebSimulation);
    connect(actions->actionSaveSimulation, &QAction::triggered, this, &ActionController::onSaveSimulation);
	connect(actions->actionLoadSimulation, &QAction::triggered, this, &ActionController::onLoadSimulation);
	connect(actions->actionComputationSettings, &QAction::triggered, this, &ActionController::onConfigureGrid);
	connect(actions->actionRunSimulation, &QAction::toggled, this, &ActionController::onRunClicked);
	connect(actions->actionRunStepForward, &QAction::triggered, this, &ActionController::onStepForward);
	connect(actions->actionRunStepBackward, &QAction::triggered, this, &ActionController::onStepBackward);
	connect(actions->actionSnapshot, &QAction::triggered, this, &ActionController::onMakeSnapshot);
	connect(actions->actionRestore, &QAction::triggered, this, &ActionController::onRestoreSnapshot);
    connect(actions->actionAcceleration, &QAction::triggered, this, &ActionController::onAcceleration);
    connect(actions->actionSimulationChanger, &QAction::triggered, this, &ActionController::onSimulationChanger);
    connect(actions->actionExit, &QAction::triggered, _mainView, &MainView::close);

	connect(actions->actionZoomIn, &QAction::triggered, this, &ActionController::onZoomInClicked);
	connect(actions->actionZoomOut, &QAction::triggered, this, &ActionController::onZoomOutClicked);
    connect(actions->actionDisplayLink, &QAction::triggered, this, &ActionController::onToggleDisplayLink);
    connect(actions->actionFullscreen, &QAction::toggled, this, &ActionController::onToggleFullscreen);
    connect(actions->actionGlowEffect, &QAction::toggled, this, &ActionController::onToggleGlowEffect);

    connect(actions->actionEditor, &QAction::toggled, this, &ActionController::onToggleEditorMode);
	connect(actions->actionMonitor, &QAction::toggled, this, &ActionController::onToggleMonitor);
	connect(actions->actionEditSimParameters, &QAction::triggered, this, &ActionController::onEditSimulationParameters);
	connect(actions->actionEditSymbols, &QAction::triggered, this, &ActionController::onEditSymbolTable);

	connect(actions->actionNewCell, &QAction::triggered, this, &ActionController::onNewCell);
	connect(actions->actionNewParticle, &QAction::triggered, this, &ActionController::onNewParticle);
	connect(actions->actionCopyEntity, &QAction::triggered, this, &ActionController::onCopyEntity);
	connect(actions->actionPasteEntity, &QAction::triggered, this, &ActionController::onPasteEntity);
	connect(actions->actionDeleteEntity, &QAction::triggered, this, &ActionController::onDeleteEntity);
	connect(actions->actionNewToken, &QAction::triggered, this, &ActionController::onNewToken);
	connect(actions->actionCopyToken, &QAction::triggered, this, &ActionController::onCopyToken);
	connect(actions->actionDeleteToken, &QAction::triggered, this, &ActionController::onDeleteToken);
	connect(actions->actionPasteToken, &QAction::triggered, this, &ActionController::onPasteToken);
	connect(actions->actionShowCellInfo, &QAction::toggled, this, &ActionController::onToggleCellInfo);
	connect(actions->actionCenterSelection, &QAction::toggled, this, &ActionController::onCenterSelection);
    connect(actions->actionCopyToClipboard, &QAction::triggered, this, &ActionController::onCopyToClipboard);
    connect(actions->actionPasteFromClipboard, &QAction::triggered, this, &ActionController::onPasteFromClipboard);

	connect(actions->actionNewRectangle, &QAction::triggered, this, &ActionController::onNewRectangle);
	connect(actions->actionNewHexagon, &QAction::triggered, this, &ActionController::onNewHexagon);
	connect(actions->actionNewParticles, &QAction::triggered, this, &ActionController::onNewParticles);
	connect(actions->actionLoadCol, &QAction::triggered, this, &ActionController::onLoadCollection);
	connect(actions->actionSaveCol, &QAction::triggered, this, &ActionController::onSaveCollection);
	connect(actions->actionCopyCol, &QAction::triggered, this, &ActionController::onCopyCollection);
	connect(actions->actionPasteCol, &QAction::triggered, this, &ActionController::onPasteCollection);
	connect(actions->actionDeleteSel, &QAction::triggered, this, &ActionController::onDeleteSelection);
	connect(actions->actionDeleteCol, &QAction::triggered, this, &ActionController::onDeleteCollection);
	connect(actions->actionRandomMultiplier, &QAction::triggered, this, &ActionController::onRandomMultiplier);
	connect(actions->actionGridMultiplier, &QAction::triggered, this, &ActionController::onGridMultiplier);

    connect(actions->actionMostFrequentCluster, &QAction::triggered, this, &ActionController::onMostFrequentCluster);

	connect(actions->actionAbout, &QAction::triggered, this, &ActionController::onShowAbout);
    connect(actions->actionGettingStarted, &QAction::triggered, this, &ActionController::onToggleGettingStarted);
    connect(actions->actionDocumentation, &QAction::triggered, this, &ActionController::onShowDocumentation);

	connect(actions->actionRestrictTPS, &QAction::triggered, this, &ActionController::onToggleRestrictTPS);
}

void ActionController::close()
{
    auto const actions = _model->getActionHolder();
    actions->actionWebSimulation->setChecked(false);
}

ActionHolder * ActionController::getActionHolder()
{
	return _model->getActionHolder();
}

void ActionController::onRunClicked(bool toggled)
{
	auto actions = _model->getActionHolder();
	if (toggled) {
		actions->actionRunStepForward->setEnabled(false);
	}
	else {
		actions->actionRunStepForward->setEnabled(true);
	}
	actions->actionRunStepBackward->setEnabled(false);

	_mainController->onRunSimulation(toggled);
}

void ActionController::onStepForward()
{
	_mainController->onStepForward();
	_model->getActionHolder()->actionRunStepBackward->setEnabled(true);
}

void ActionController::onStepBackward()
{
	bool emptyStack = false;
	_mainController->onStepBackward(emptyStack);
	if (emptyStack) {
		_model->getActionHolder()->actionRunStepBackward->setEnabled(false);
	}
	_simulationViewWidget->refresh();
}

void ActionController::onMakeSnapshot()
{
	_mainController->onMakeSnapshot();
	_model->getActionHolder()->actionRestore->setEnabled(true);
}

void ActionController::onRestoreSnapshot()
{
	_mainController->onRestoreSnapshot();
	_simulationViewWidget->refresh();
}

void ActionController::onAcceleration(bool toggled)
{
    auto parameters = _mainModel->getExecutionParameters();
    parameters.activateFreezing = toggled;
    if (toggled) {
        bool ok;
        auto const accelerationTimesteps = QInputDialog::getInt(
            _mainView, "Acceleration", "Enter the number of frames of acceleration", parameters.freezingTimesteps, 1, 100, 1, &ok);
        if (!ok) {
            auto const actionHolder = _model->getActionHolder();
            actionHolder->actionAcceleration->setChecked(false);
            return;
        }
        parameters.freezingTimesteps = accelerationTimesteps;
    }
    _mainModel->setExecutionParameters(parameters);
    _mainController->onUpdateExecutionParameters(parameters);
}

void ActionController::onSimulationChanger(bool toggled)
{
    _mainController->onSimulationChanger(toggled);
}

void ActionController::onZoomInClicked()
{
    auto zoomFactor = _simulationViewWidget->getZoomFactor();
	_simulationViewWidget->setZoomFactor(zoomFactor * 2);

    if(!_model->isEditMode()) {
        if (_simulationViewWidget->getZoomFactor() > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
            _model->getActionHolder()->actionEditor->toggle();
        }
        else {
            setPixelOrVectorView();
        }
    }
    _simulationViewWidget->refresh();
    updateActionsEnableState();
}

void ActionController::onZoomOutClicked()
{
    auto zoomFactor = _simulationViewWidget->getZoomFactor();
    _simulationViewWidget->setZoomFactor(zoomFactor / 2);

    if (_model->isEditMode()) {
        if (_simulationViewWidget->getZoomFactor() > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
        }
        else {
            _model->getActionHolder()->actionEditor->toggle();
        }
    }
    else {
        setPixelOrVectorView();
    }
    _simulationViewWidget->refresh();
    updateActionsEnableState();
}

void ActionController::onToggleDisplayLink(bool toggled)
{
    _mainController->onDisplayLink(toggled);
}

void ActionController::onToggleFullscreen(bool toogled)
{
	Qt::WindowStates state = _mainView->windowState();
	if (toogled) {
		state |= Qt::WindowFullScreen;
	}
	else {
		state &= ~Qt::WindowFullScreen;
	}
	_mainView->setWindowState(state);

	GuiSettings::setSettingsValue(Const::MainViewFullScreenKey, toogled);
}

void ActionController::onToggleGlowEffect(bool toogled)
{
    auto parameters = _mainModel->getExecutionParameters();
    parameters.imageGlow = toogled;
    _mainModel->setExecutionParameters(parameters);
    _mainController->onUpdateExecutionParameters(parameters);
    _mainView->refresh();
}

void ActionController::onToggleEditorMode(bool toggled)
{
    _model->setEditMode(toggled);
    if (toggled) {
        _infoController->setRendering(InfoController::Rendering::Item);
        _simulationViewWidget->disconnectView();
        _simulationViewWidget->setActiveScene(ActiveView::ItemScene);
        _simulationViewWidget->connectView();
    }
	else {
        setPixelOrVectorView();
	}
    _simulationViewWidget->refresh();
    updateActionsEnableState();

	Q_EMIT _toolbar->getContext()->show(toggled);
	Q_EMIT _dataEditor->getContext()->show(toggled);
}

void ActionController::onToggleMonitor(bool toggled)
{
	_monitor->onShow(toggled);
}

void ActionController::onNewSimulation()
{
	NewSimulationDialog dialog(_mainModel->getSimulationParameters(), _mainModel->getSymbolTable(), _serializer, _mainView);
	if (dialog.exec()) {
		_mainController->onNewSimulation(*dialog.getConfig(), *dialog.getEnergy());

		settingUpNewSimulation(_mainController->getSimulationConfig());
	}
}

void ActionController::onWebSimulation(bool toogled)
{
    if (toogled) {
        if (!_webSimController->onConnectToSimulation()) {
            auto actions = _model->getActionHolder();
            actions->actionWebSimulation->setChecked(false);
        }
    }
    else {
        auto const currentSimulationId = _webSimController->getCurrentSimulationId();
        auto const currentToken = _webSimController->getCurrentToken();
        if (currentSimulationId && currentToken) {
            _webSimController->onDisconnectToSimulation(*currentSimulationId, *currentToken);
        }
    }
}

void ActionController::onSaveSimulation()
{
	QString filename = QFileDialog::getSaveFileName(_mainView, "Save Simulation", "", "Alien Simulation(*.sim)");
	if (!filename.isEmpty()) {
		_mainController->onSaveSimulation(filename.toStdString());
	}
}

void ActionController::onLoadSimulation()
{
	QString filename = QFileDialog::getOpenFileName(_mainView, "Load Simulation", "", "Alien Simulation (*.sim)");
	if (!filename.isEmpty()) {
		if (_mainController->onLoadSimulation(filename.toStdString(), MainController::LoadOption::SaveOldSim)) {
			settingUpNewSimulation(_mainController->getSimulationConfig());
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorLoadSimulation);
			msgBox.exec();
		}
	}
}

void ActionController::onConfigureGrid()
{
	ComputationSettingsDialog dialog(_mainController->getSimulationConfig(), _mainView);
    if (dialog.exec()) {

        auto config = _mainController->getSimulationConfig();
        config->universeSize = *dialog.getUniverseSize();
        config->cudaConstants = *dialog.getCudaConstants();

        auto const extrapolateContent = *dialog.isExtrapolateContent();
        _mainController->onRecreateUniverse(config, extrapolateContent);
        settingUpNewSimulation(config);
	}
}

void ActionController::onEditSimulationParameters()
{
    auto const config = _mainController->getSimulationConfig();
    SimulationParametersDialog dialog(config->parameters, _serializer, _mainView);
	if (dialog.exec()) {
        auto const parameters = dialog.getSimulationParameters();
		_mainModel->setSimulationParameters(parameters);
		_mainController->onUpdateSimulationParameters(parameters);
	}
}

void ActionController::onEditSymbolTable()
{
	auto origSymbols = _mainModel->getSymbolTable();
	SymbolTableDialog dialog(origSymbols->clone(), _serializer, _mainView);
	if (dialog.exec()) {
		origSymbols->getSymbolsFrom(dialog.getSymbolTable());
		Q_EMIT _dataEditor->getContext()->refresh();
	}
}

void ActionController::onNewCell()
{
	_repository->addAndSelectCell(_model->getPositionDeltaForNewEntity());
	_repository->reconnectSelectedCells();
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onNewParticle()
{
	_repository->addAndSelectParticle(_model->getPositionDeltaForNewEntity());
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onCopyEntity()
{
	auto const& selectedCellIds = _repository->getSelectedCellIds();
	auto const& selectedParticleIds = _repository->getSelectedParticleIds();
	if (!selectedCellIds.empty()) {
		CHECK(selectedParticleIds.empty());
		auto const& cell = _repository->getCellDescRef(*selectedCellIds.begin());
		auto const& cluster = _repository->getClusterDescRef(*selectedCellIds.begin());
		QVector2D vel = Physics::tangentialVelocity(*cell.pos - *cluster.pos, { *cluster.vel, *cluster.angularVel });
		_model->setCellCopied(cell, vel);
	}
	if (!selectedParticleIds.empty()) {
		CHECK(selectedCellIds.empty());
		auto const& particle = _repository->getParticleDescRef(*selectedParticleIds.begin());
		_model->setParticleCopied(particle);
	}
	updateActionsEnableState();
}

void ActionController::onLoadCollection()
{
	QString filename = QFileDialog::getOpenFileName(_mainView, "Load Collection", "", "Alien Collection (*.aco)");
	if (!filename.isEmpty()) {
		DataDescription desc;
		if (SerializationHelper::loadFromFile<DataDescription>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeDataDescription(data); }, desc)) {
			_repository->addAndSelectData(desc, _model->getPositionDeltaForNewEntity());
			Q_EMIT _notifier->notifyDataRepositoryChanged({
				Receiver::DataEditor,
				Receiver::Simulation,
				Receiver::VisualEditor,
				Receiver::ActionController
			}, UpdateDescription::All);
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorLoadCollection);
			msgBox.exec();
		}
	}
}

void ActionController::onSaveCollection()
{
	QString filename = QFileDialog::getSaveFileName(_mainView, "Save Collection", "", "Alien Collection (*.aco)");
	if (!filename.isEmpty()) {
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeDataDescription(_repository->getExtendedSelection()); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorSaveCollection);
			msgBox.exec();
			return;
		}
	}
}

void ActionController::onCopyCollection()
{
	DataDescription copiedData = _repository->getExtendedSelection();
	_model->setCopiedCollection(copiedData);
	updateActionsEnableState();
}

void ActionController::onPasteCollection()
{
	DataDescription copiedData = _model->getCopiedCollection();
	_repository->addAndSelectData(copiedData, _model->getPositionDeltaForNewEntity());
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor,Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onDeleteSelection()
{
	_repository->deleteSelection();
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor, Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onDeleteCollection()
{
	_repository->deleteExtendedSelection();
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor, Receiver::ActionController
	}, UpdateDescription::All);
}

namespace
{
	void modifyDescription(DataDescription& data, QVector2D const& posDelta, optional<double> const& velocityXDelta
		, optional<double> const& velocityYDelta, optional<double> const& angularVelocityDelta)
	{
		if (data.clusters) {
			for (auto& cluster : data.clusters.get()) {
				*cluster.pos += posDelta;
				if (velocityXDelta) {
					cluster.vel->setX(cluster.vel->x() + *velocityXDelta);
				}
				if (velocityYDelta) {
					cluster.vel->setY(cluster.vel->y() + *velocityYDelta);
				}
				if (angularVelocityDelta) {
					*cluster.angularVel += *angularVelocityDelta;
				}
				if (cluster.cells) {
					for (auto& cell : cluster.cells.get()) {
						*cell.pos += posDelta;
					}
				}
			}
		}
		if (data.particles) {
			for (auto& particle : data.particles.get()) {
				*particle.pos += posDelta;
				if (velocityXDelta) {
					particle.vel->setX(particle.vel->x() + *velocityXDelta);
				}
				if (velocityYDelta) {
					particle.vel->setY(particle.vel->y() + *velocityYDelta);
				}
			}
		}
	}
}

void ActionController::onRandomMultiplier()
{
	RandomMultiplierDialog dialog;
	if (dialog.exec()) {
		DataDescription data = _repository->getExtendedSelection();
		IntVector2D universeSize = _mainController->getSimulationConfig()->universeSize;
		vector<DataRepository::DataAndAngle> addedData;
		for (int i = 0; i < dialog.getNumberOfCopies(); ++i) {
			DataDescription dataCopied = data;
			QVector2D posDelta(_numberGenerator->getRandomReal(0.0, universeSize.x), _numberGenerator->getRandomReal(0.0, universeSize.y));
			optional<double> velocityX;
			optional<double> velocityY;
			optional<double> angle;
			optional<double> angularVelocity;
			if (dialog.isChangeVelX()) {
				velocityX = _numberGenerator->getRandomReal(dialog.getVelXMin(), dialog.getVelXMax());
			}
			if (dialog.isChangeVelY()) {
				velocityY = _numberGenerator->getRandomReal(dialog.getVelYMin(), dialog.getVelYMax());
			}
			if (dialog.isChangeAngle()) {
				angle = _numberGenerator->getRandomReal(dialog.getAngleMin(), dialog.getAngleMax());
			}
			if (dialog.isChangeAngVel()) {
				angularVelocity = _numberGenerator->getRandomReal(dialog.getAngVelMin(), dialog.getAngVelMax());
			}
			modifyDescription(dataCopied, posDelta, velocityX, velocityY, angularVelocity);
			addedData.emplace_back(DataRepository::DataAndAngle{ dataCopied, angle });
		}
		_repository->addDataAtFixedPosition(addedData);
		Q_EMIT _notifier->notifyDataRepositoryChanged({
			Receiver::DataEditor,
			Receiver::Simulation,
			Receiver::VisualEditor,
			Receiver::ActionController
		}, UpdateDescription::All);
	}
}

void ActionController::onGridMultiplier()
{
	DataDescription data = _repository->getExtendedSelection();
	QVector2D center = data.calcCenter();
	GridMultiplierDialog dialog(center);
	if (dialog.exec()) {
		QVector2D initialDelta(dialog.getInitialPosX(), dialog.getInitialPosY());
		initialDelta -= center;
		vector<DataRepository::DataAndAngle> addedData;
		for (int i = 0; i < dialog.getHorizontalNumber(); ++i) {
			for (int j = 0; j < dialog.getVerticalNumber(); ++j) {
				if (i == 0 && j == 0 && initialDelta.lengthSquared() < FLOATINGPOINT_MEDIUM_PRECISION) {
					continue;
				}
				DataDescription dataCopied = data;
				optional<double> velocityX;
				optional<double> velocityY;
				optional<double> angle;
				optional<double> angularVelocity;
				if (dialog.isChangeAngle()) {
					angle = dialog.getInitialAngle() + i*dialog.getHorizontalAngleIncrement() + j*dialog.getVerticalAngleIncrement();
				}
				if (dialog.isChangeVelocityX()) {
					velocityX = dialog.getInitialVelX() + i*dialog.getHorizontalVelocityXIncrement() + j*dialog.getVerticalVelocityXIncrement();
				}
				if (dialog.isChangeVelocityY()) {
					velocityY = dialog.getInitialVelY() + j*dialog.getHorizontalVelocityYIncrement() + j*dialog.getVerticalVelocityYIncrement();
				}
				if (dialog.isChangeAngularVelocity()) {
					angularVelocity = dialog.getInitialAngVel() + i*dialog.getHorizontalAngularVelocityIncrement() + j*dialog.getVerticalAngularVelocityIncrement();
				}

				QVector2D posDelta(i*dialog.getHorizontalInterval(), j*dialog.getVerticalInterval());
				posDelta += initialDelta;

				modifyDescription(dataCopied, posDelta, velocityX, velocityY, angularVelocity);
				addedData.emplace_back(DataRepository::DataAndAngle{ dataCopied, angle });
			}
		}
		_repository->addDataAtFixedPosition(addedData);
		Q_EMIT _notifier->notifyDataRepositoryChanged({
			Receiver::DataEditor,
			Receiver::Simulation,
			Receiver::VisualEditor,
			Receiver::ActionController
		}, UpdateDescription::All);
	}
}

void ActionController::onMostFrequentCluster()
{
    _mainController->onAddMostFrequentClusterToSimulation();
}

void ActionController::onDeleteEntity()
{
	onDeleteSelection();
}

void ActionController::onPasteEntity()
{
	DataDescription copiedData = _model->getCopiedEntity();
	_repository->addAndSelectData(copiedData, _model->getPositionDeltaForNewEntity());
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor,Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onNewToken()
{
	_repository->addToken();
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor, Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onCopyToken()
{
	auto cellIds = _repository->getSelectedCellIds();
	CHECK(cellIds.size() == 1);
	auto tokenIndex = _repository->getSelectedTokenIndex();
	CHECK(tokenIndex);
	auto const& cell = _repository->getCellDescRef(*cellIds.begin());
	auto const& token = cell.tokens->at(*tokenIndex);

	_model->setCopiedToken(token);
	updateActionsEnableState();
}

void ActionController::onPasteToken()
{ 
	auto const& token = _model->getCopiedToken();
	_repository->addToken(token);
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor, Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onDeleteToken()
{
	_repository->deleteToken();
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor, Receiver::ActionController
	}, UpdateDescription::All);
}

void ActionController::onToggleCellInfo(bool show)
{
	Q_EMIT _notifier->toggleCellInfo(show);
}

void ActionController::onCenterSelection(bool centerSelection)
{
	_simulationViewWidget->toggleCenterSelection(centerSelection);
    _simulationViewWidget->refresh();
}

void ActionController::onCopyToClipboard()
{
    auto const cellIds = _repository->getSelectedCellIds();
    CHECK(cellIds.size() == 1);
    auto const tokenIndex = _repository->getSelectedTokenIndex();
    CHECK(tokenIndex);
    auto const& cell = _repository->getCellDescRef(*cellIds.begin());
    auto const& token = cell.tokens->at(*tokenIndex);

    auto const& tokenMemory = *token.data;
    auto tokenMemoryInHex = tokenMemory.toHex();

    for (int index = 255 * 2; index > 0; index -= 2) {
        tokenMemoryInHex.insert(index, char(' '));
    }
    
    auto clipboard = QApplication::clipboard();
    clipboard->setText(QString::fromStdString(tokenMemoryInHex.toStdString()));
}

void ActionController::onPasteFromClipboard()
{
    auto const cellIds = _repository->getSelectedCellIds();
    CHECK(cellIds.size() == 1);
    auto const tokenIndex = _repository->getSelectedTokenIndex();
    CHECK(tokenIndex);
    auto& cell = _repository->getCellDescRef(*cellIds.begin());
    auto& token = cell.tokens->at(*tokenIndex);

    auto& tokenMemory = *token.data;
    auto clipboard = QApplication::clipboard();
    QString newTokenMemoryInHex = clipboard->text();
    newTokenMemoryInHex.remove(QChar(' '));
    tokenMemory = QByteArray::fromHex(newTokenMemoryInHex.toUtf8());

    auto const tokenMemorySize = _mainModel->getSimulationParameters().tokenMemorySize;
    if (tokenMemorySize != tokenMemory.size()) {
        QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorPasteFromClipboard);
        msgBox.exec();
        return;
    }
    Q_EMIT _notifier->notifyDataRepositoryChanged({
        Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor, Receiver::ActionController
    }, UpdateDescription::All);
}

void ActionController::onNewRectangle()
{
	NewRectangleDialog dialog(_mainModel->getSimulationParameters());
	if (dialog.exec()) {
		IntVector2D size = dialog.getBlockSize();
		double distance = dialog.getDistance();
		double energy = dialog.getInternalEnergy();
        int colorCode = dialog.getColorCode();

		uint64_t id = 0;

		vector<vector<CellDescription>> cellMatrix;
		for (int x = 0; x < size.x; ++x) {
			vector<CellDescription> cellRow;
			for (int y = 0; y < size.y; ++y) {
				int maxConn = 4;
				if (x == 0 || x == size.x - 1) {
					--maxConn;
				}
				if (y == 0 || y == size.y - 1) {
					--maxConn;
				}
				cellRow.push_back(CellDescription().setId(++id).setEnergy(energy)
					.setPos({ static_cast<float>(x), static_cast<float>(y) })
					.setMaxConnections(maxConn).setFlagTokenBlocked(false)
					.setTokenBranchNumber(0).setMetadata(CellMetadata().setColor(colorCode))
					.setCellFeature(CellFeatureDescription()));
			}
			cellMatrix.push_back(cellRow);
		}
		for (int x = 0; x < size.x; ++x) {
			for (int y = 0; y < size.y; ++y) {
				if (x < size.x - 1) {
					cellMatrix[x][y].addConnection(cellMatrix[x + 1][y].id);
				}
				if (x > 0) {
					cellMatrix[x][y].addConnection(cellMatrix[x - 1][y].id);
				}
				if (y < size.y - 1) {
					cellMatrix[x][y].addConnection(cellMatrix[x][y + 1].id);
				}
				if (y > 0) {
					cellMatrix[x][y].addConnection(cellMatrix[x][y - 1].id);
				}
			}
		}

		auto cluster = ClusterDescription().setPos({ static_cast<float>(size.x) / 2.0f - 0.5f, static_cast<float>(size.y) / 2.0f - 0.5f })
			.setVel({ 0, 0 })
			.setAngle(0).setAngularVel(0).setMetadata(ClusterMetadata());
		for (int x = 0; x < size.x; ++x) {
			for (int y = 0; y < size.y; ++y) {
				cluster.addCell(cellMatrix[x][y]);
			}
		}

		_repository->addAndSelectData(DataDescription().addCluster(cluster), { 0, 0 });
		Q_EMIT _notifier->notifyDataRepositoryChanged({
			Receiver::DataEditor,
			Receiver::Simulation,
			Receiver::VisualEditor,
			Receiver::ActionController
		}, UpdateDescription::All);
	}
}

namespace
{
}

void ActionController::onNewHexagon()
{
	NewHexagonDialog dialog(_mainModel->getSimulationParameters());
	if (dialog.exec()) {

		int layers = dialog.getLayers();
		double dist = dialog.getDistance();
		double energy = dialog.getCellEnergy();
        int colorCode = dialog.getColorCode();

        auto const factory = ServiceLocator::getInstance().getService<DescriptionFactory>();
        auto hexagon = factory->createHexagon(
            DescriptionFactory::CreateHexagonParameters().layers(layers).cellDistance(dist).cellEnergy(energy).colorCode(colorCode));

		_repository->addAndSelectData(DataDescription().addCluster(hexagon), { 0, 0 });
		Q_EMIT _notifier->notifyDataRepositoryChanged({
			Receiver::DataEditor,
			Receiver::Simulation,
			Receiver::VisualEditor,
			Receiver::ActionController
		}, UpdateDescription::All);
	}
}

void ActionController::onNewParticles()
{
	NewParticlesDialog dialog;
	if (dialog.exec()) {
		double totalEnergy = dialog.getTotalEnergy();
		double maxEnergyPerParticle = dialog.getMaxEnergyPerParticle();

		_repository->addRandomParticles(totalEnergy, maxEnergyPerParticle);
		Q_EMIT _notifier->notifyDataRepositoryChanged({
			Receiver::DataEditor,
			Receiver::Simulation,
			Receiver::VisualEditor,
			Receiver::ActionController
		}, UpdateDescription::All);
	}
}

void ActionController::onShowAbout()
{
    QFile file("://Version.txt");
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(nullptr, "error", file.errorString());
    }
    else {
        QTextStream in(&file);
        auto version = in.readLine();
        QMessageBox msgBox(QMessageBox::Information, "About", QString(Const::InfoAbout).arg(version));
        msgBox.exec();
    }
}

void ActionController::onToggleGettingStarted(bool toggled)
{
    _mainView->toggleGettingStarted(toggled);
}

void ActionController::onShowDocumentation()
{
	_mainView->showDocumentation();
}

void ActionController::onToggleRestrictTPS(bool toggled)
{
	if (toggled) {
        bool ok;
        auto restrictTPS = QInputDialog::getInt(
            _mainView,
            "Restrict TPS",
            "Enter the time steps per seconds",
            50,
            1,
            1000,
            1,
            &ok);
        if (!ok) {
            auto const actionHolder = _model->getActionHolder();
            actionHolder->actionRestrictTPS->setChecked(false);
            return;
        }
        _mainController->onRestrictTPS(restrictTPS);
	}
	else {
		_mainController->onRestrictTPS(boost::none);
	}
}

void ActionController::receivedNotifications(set<Receiver> const & targets)
{
	if (targets.find(Receiver::ActionController) == targets.end()) {
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
			freeTokenOfSelectedCell = _mainModel->getSimulationParameters().cellMaxToken - tokenOfSelectedCell;
		}
	}

	_model->setEntitySelected(selectedCells == 1 || selectedParticles == 1);
	_model->setCellWithTokenSelected(tokenOfSelectedCell > 0);
	_model->setCellWithFreeTokenSelected(freeTokenOfSelectedCell > 0);
	_model->setCollectionSelected(selectedCells > 0 || selectedParticles > 0);

	updateActionsEnableState();
}

void ActionController::settingUpNewSimulation(SimulationConfig const& config)
{
    auto actions = _model->getActionHolder();
    actions->actionRunSimulation->setChecked(false);
    actions->actionRestore->setEnabled(false);
    actions->actionRunStepBackward->setEnabled(false);
    actions->actionDisplayLink->setEnabled(true);
    actions->actionDisplayLink->setChecked(true);
    actions->actionGlowEffect->setEnabled(true);
    actions->actionSimulationChanger->setChecked(false);
    actions->actionWebSimulation->setChecked(false);
    onRunClicked(false);
    onToggleCellInfo(actions->actionShowCellInfo->isChecked());
    onToggleRestrictTPS(actions->actionRestrictTPS->isChecked());
}

void ActionController::updateActionsEnableState()
{
	bool editMode = _model->isEditMode();
	bool entitySelected = _model->isEntitySelected();
	bool entityCopied = _model->isEntityCopied();
	bool cellWithTokenSelected = _model->isCellWithTokenSelected();
	bool cellWithFreeTokenSelected = _model->isCellWithFreeTokenSelected();
	bool tokenCopied = _model->isTokenCopied();
	bool collectionSelected = _model->isCollectionSelected();
	bool collectionCopied = _model->isCollectionCopied();

	auto actions = _model->getActionHolder();
    actions->actionEditor->setEnabled(_simulationViewWidget->getZoomFactor() > Const::MinZoomLevelForEditor - FLOATINGPOINT_MEDIUM_PRECISION);
    actions->actionGlowEffect->setEnabled(!editMode);
	actions->actionShowCellInfo->setEnabled(editMode);
    actions->actionCenterSelection->setEnabled(editMode);

	actions->actionNewCell->setEnabled(true);
	actions->actionNewParticle->setEnabled(true);
	actions->actionCopyEntity->setEnabled(editMode && entitySelected);
	actions->actionPasteEntity->setEnabled(editMode && entityCopied);
	actions->actionDeleteEntity->setEnabled(editMode && entitySelected);
	actions->actionNewToken->setEnabled(editMode && entitySelected);
	actions->actionCopyToken->setEnabled(editMode && cellWithTokenSelected);
	actions->actionPasteToken->setEnabled(editMode && entitySelected && tokenCopied);
	actions->actionDeleteToken->setEnabled(editMode && cellWithTokenSelected);
    actions->actionCopyToClipboard ->setEnabled(editMode && cellWithTokenSelected);
    actions->actionPasteFromClipboard->setEnabled(editMode && cellWithTokenSelected);

	actions->actionNewRectangle->setEnabled(true);
	actions->actionNewHexagon->setEnabled(true);
	actions->actionNewParticles->setEnabled(true);
	actions->actionLoadCol->setEnabled(true);
	actions->actionSaveCol->setEnabled(editMode && collectionSelected);
	actions->actionCopyCol->setEnabled(editMode && collectionSelected);
	actions->actionPasteCol->setEnabled(collectionCopied);
	actions->actionDeleteSel->setEnabled(editMode && collectionSelected);
	actions->actionDeleteCol->setEnabled(editMode && collectionSelected);
	actions->actionRandomMultiplier->setEnabled(collectionSelected);
	actions->actionGridMultiplier->setEnabled(collectionSelected);
}

void ActionController::setPixelOrVectorView()
{
    if (_simulationViewWidget->getZoomFactor() > Const::ZoomLevelForAutomaticVectorViewSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
        if (ActiveView::VectorScene != _simulationViewWidget->getActiveView()) {
            _infoController->setRendering(InfoController::Rendering::Vector);
            _simulationViewWidget->disconnectView();
            _simulationViewWidget->setActiveScene(ActiveView::VectorScene);
            _simulationViewWidget->connectView();
        }
    }
    else {
        if (ActiveView::PixelScene != _simulationViewWidget->getActiveView()) {
            _infoController->setRendering(InfoController::Rendering::Pixel);
            _simulationViewWidget->disconnectView();
            _simulationViewWidget->setActiveScene(ActiveView::PixelScene);
            _simulationViewWidget->connectView();
        }
    }
}
