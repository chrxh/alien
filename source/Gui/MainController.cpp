#include <iostream>
#include <fstream>

#include <QTime>
#include <QTimer>
#include <QProgressDialog>
#include <QCoreApplication>
#include <QMessageBox>

#include "Base/GlobalFactory.h"
#include "Base/ServiceLocator.h"

#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/DescriptionHelper.h"
#include "ModelBasic/SimulationMonitor.h"
#include "ModelBasic/SerializationHelper.h"
#include "ModelBasic/SimulationChanger.h"

#include "ModelGpu/SimulationAccessGpu.h"
#include "ModelGpu/SimulationControllerGpu.h"
#include "ModelGpu/ModelGpuBuilderFacade.h"
#include "ModelGpu/ModelGpuData.h"
#include "ModelGpu/SimulationMonitorGpu.h"

#include "Web/WebAccess.h"
#include "Web/WebBuilderFacade.h"

#include "MessageHelper.h"
#include "VersionController.h"
#include "InfoController.h"
#include "MainController.h"
#include "MainView.h"
#include "MainModel.h"
#include "DataRepository.h"
#include "Notifier.h"
#include "SimulationConfig.h"
#include "DataAnalyzer.h"
#include "QApplicationHelper.h"
#include "Queue.h"
#include "WebSimulationController.h"

namespace Const
{
    std::string const AutoSaveFilename = "autosave.sim";
    std::string const AutoSaveForLoadingFilename = "autosave_load.sim";
}

MainController::MainController(QObject * parent)
	: QObject(parent)
{
}

MainController::~MainController()
{
    delete _view;
}

void MainController::init()
{
    _model = new MainModel(this);
    _view = new MainView();

    _controllerBuildFunc = [](int typeId, IntVector2D const& universeSize, SymbolTable* symbols,
        SimulationParameters const& parameters, map<string, int> const& typeSpecificData, uint timestepAtBeginning) -> SimulationController*
    {
        if (ModelComputationType(typeId) == ModelComputationType::Gpu) {
            auto facade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
            ModelGpuData data(typeSpecificData);
            return facade->buildSimulationController({ universeSize, symbols, parameters }, data, timestepAtBeginning);
        }
        else {
            THROW_NOT_IMPLEMENTED();
        }
    };
    _accessBuildFunc = [](SimulationController* controller) -> SimulationAccess*
    {
        if (auto controllerGpu = dynamic_cast<SimulationControllerGpu*>(controller)) {
            auto modelGpuFacade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
            SimulationAccessGpu* access = modelGpuFacade->buildSimulationAccess();
            access->init(controllerGpu);
            return access;
        }
        else {
            THROW_NOT_IMPLEMENTED();
        }
    };
    _monitorBuildFunc = [](SimulationController* controller) -> SimulationMonitor*
    {
        if (auto controllerGpu = dynamic_cast<SimulationControllerGpu*>(controller)) {
            auto facade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
            SimulationMonitorGpu* moni = facade->buildSimulationMonitor();
            moni->init(controllerGpu);
            return moni;
        }
        else {
            THROW_NOT_IMPLEMENTED();
        }
    };

    auto modelBasicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
    auto serializer = modelBasicFacade->buildSerializer();
    auto descHelper = modelBasicFacade->buildDescriptionHelper();
    auto versionController = new VersionController();
    SET_CHILD(_serializer, serializer);
    SET_CHILD(_descHelper, descHelper);
    SET_CHILD(_versionController, versionController);
    _repository = new DataRepository(this);
    _notifier = new Notifier(this);
    _dataAnalyzer = new DataAnalyzer(this);
    auto worker = new Queue(this);
    SET_CHILD(_worker, worker);

    auto webFacade = ServiceLocator::getInstance().getService<WebBuilderFacade>();
    auto webAccess = webFacade->buildWebController();
    SET_CHILD(_webAccess, webAccess);

    auto webSimController = new WebSimulationController(webAccess, _view);
    SET_CHILD(_webSimController, webSimController);

    _serializer->init(_controllerBuildFunc, _accessBuildFunc);
    _view->init(_model, this, _serializer, _repository, _notifier, _webSimController);
    _worker->init(_serializer);

    QApplicationHelper::processEventsForMilliSec(1000);

    if (!onLoadSimulation(getPathToApp() + Const::AutoSaveFilename, LoadOption::Non)) {

        //default simulation
        auto const modelGpuFacade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();

        auto config = boost::make_shared<_SimulationConfig>();
        config->cudaConstants = modelGpuFacade->getDefaultCudaConstants();
        config->universeSize = IntVector2D({ 2000 , 1000 });
        config->symbolTable = modelBasicFacade->getDefaultSymbolTable();
        config->parameters = modelBasicFacade->getDefaultSimulationParameters();
        onNewSimulation(config, 0);
    }

    auto config = getSimulationConfig();
    _view->getInfoController()->setDevice(InfoController::Device::Gpu);

    _view->initGettingStartedWindow();

    //auto save every 20 min
    _autosaveTimer = new QTimer(this);
    connect(_autosaveTimer, &QTimer::timeout, this, (void(MainController::*)())(&MainController::autoSave));
    _autosaveTimer->start(1000 * 60 * 20);
}

void MainController::autoSave()
{
    auto progress = MessageHelper::createProgressDialog("Autosaving...", _view);
    autoSaveIntern(getPathToApp() + Const::AutoSaveFilename);
    delete progress;
}

void MainController::serializeSimulationAndWaitUntilFinished()
{
    QEventLoop pause;
    bool finished = false;
    auto connection = _serializer->connect(_serializer, &Serializer::serializationFinished, [&]() {
        finished = true;
        pause.quit();
    });
    if (dynamic_cast<SimulationControllerGpu*>(_simController)) {
        _serializer->serialize(_simController, int(ModelComputationType::Gpu));
    }
    else {
        THROW_NOT_IMPLEMENTED();
    }
    while (!finished) {
        pause.exec();
    }
    QObject::disconnect(connection);
}

void MainController::autoSaveIntern(std::string const& filename)
{
    saveSimulationIntern(filename);
	QApplicationHelper::processEventsForMilliSec(1000);
}

void MainController::saveSimulationIntern(string const & filename)
{
    serializeSimulationAndWaitUntilFinished();
    SerializationHelper::saveToFile(filename, [&]() { return _serializer->retrieveSerializedSimulation(); });
}

string MainController::getPathToApp() const
{
    auto result = qApp->applicationDirPath();
    if (!result.endsWith("/") && !result.endsWith("\\")) {
        result += "/";
    }
    return result.toStdString();
}

void MainController::onRunSimulation(bool run)
{
	_simController->setRun(run);
	_versionController->clearStack();
}

void MainController::onStepForward()
{
	_versionController->saveSimulationContentToStack();
	_simController->calculateSingleTimestep();
}

void MainController::onStepBackward(bool& emptyStack)
{
	_versionController->loadSimulationContentFromStack();
	emptyStack = _versionController->isStackEmpty();
    Q_EMIT _notifier->notifyDataRepositoryChanged({
        Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor,Receiver::ActionController
    }, UpdateDescription::All);
}

void MainController::onMakeSnapshot()
{
	_versionController->makeSnapshot();
}

void MainController::onRestoreSnapshot()
{
	_versionController->restoreSnapshot();
    Q_EMIT _notifier->notifyDataRepositoryChanged({
        Receiver::DataEditor, Receiver::VisualEditor,Receiver::ActionController
    }, UpdateDescription::All);
}

void MainController::onDisplayLink(bool toggled)
{
    _simController->setEnableCalculateFrames(toggled);
}

void MainController::onSimulationChanger(bool toggled)
{
    if (toggled) {
        auto parameters = _simController->getContext()->getSimulationParameters();
        _simChanger->activate(parameters);
    }
    else {
        _simChanger->deactivate();
    }
}

void MainController::initSimulation(SymbolTable* symbolTable, SimulationParameters const& parameters)
{
    auto const modelBasicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();

	_model->setSimulationParameters(parameters);
    _model->setExecutionParameters(modelBasicFacade->getDefaultExecutionParameters());
	_model->setSymbolTable(symbolTable);

	connectSimController();

	auto context = _simController->getContext();
	_descHelper->init(context);
	_versionController->init(_simController->getContext(), _accessBuildFunc(_simController));
	_repository->init(_notifier, _accessBuildFunc(_simController), _descHelper, context);
    _dataAnalyzer->init(_accessBuildFunc(_simController), _repository, _notifier);

	auto simMonitor = _monitorBuildFunc(_simController);
	SET_CHILD(_simMonitor, simMonitor);

    auto webSimMonitor = _monitorBuildFunc(_simController);
    auto space = context->getSpaceProperties();
    _webSimController->init(_accessBuildFunc(_simController), webSimMonitor, getSimulationConfig());

    auto simChanger = modelBasicFacade->buildSimulationChanger(simMonitor, context->getNumberGenerator());
    for (auto const& connection : _simChangerConnections) {
        QObject::disconnect(connection);
    }
    SET_CHILD(_simChanger, simChanger);
    _simChangerConnections.emplace_back(connect(
        _simController,
        &SimulationController::nextTimestepCalculated,
        _simChanger,
        &SimulationChanger::notifyNextTimestep));
    _simChangerConnections.emplace_back(connect(_simChanger, &SimulationChanger::simulationParametersChanged, [&] {
        auto newParameters = _simChanger->retrieveSimulationParameters();
        onUpdateSimulationParameters(newParameters);
    }));

    SimulationAccess* accessForWidgets;
	_view->setupEditors(_simController, _accessBuildFunc(_simController));
}

void MainController::recreateSimulation(string const & serializedSimulation)
{
	delete _simController;
	_simController = _serializer->deserializeSimulation(serializedSimulation);

	auto symbolTable = _simController->getContext()->getSymbolTable();
	auto simulationParameters = _simController->getContext()->getSimulationParameters();

	initSimulation(symbolTable, simulationParameters);

	_view->refresh();
}

void MainController::onNewSimulation(SimulationConfig const& config, double energyAtBeginning)
{
	delete _simController;
	auto facade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
    auto simulationControllerConfig =
        ModelGpuBuilderFacade::Config{ config->universeSize, config->symbolTable, config->parameters};
    auto data = ModelGpuData(config->cudaConstants);
	_simController = facade->buildSimulationController(simulationControllerConfig, data);

	initSimulation(config->symbolTable, config->parameters);

	addRandomEnergy(energyAtBeginning);

	_view->refresh();
}

void MainController::onSaveSimulation(string const& filename)
{
    auto progress = MessageHelper::createProgressDialog("Saving...", _view);

    saveSimulationIntern(filename);

    QApplicationHelper::processEventsForMilliSec(1000);
    delete progress;
}

bool MainController::onLoadSimulation(string const & filename, LoadOption option)
{
    auto progress = MessageHelper::createProgressDialog("Loading...", _view);

    if (LoadOption::SaveOldSim == option) {
        autoSaveIntern(getPathToApp() + Const::AutoSaveForLoadingFilename);
    }
	delete _simController;
    _simController = nullptr;

    if (!SerializationHelper::loadFromFile<SimulationController*>(filename, [&](string const& data) { return _serializer->deserializeSimulation(data); }, _simController)) {

        //load old simulation
        if (LoadOption::SaveOldSim == option) {
            CHECK(SerializationHelper::loadFromFile<SimulationController*>(getPathToApp() +
                Const::AutoSaveForLoadingFilename,
                [&](string const& data) { return _serializer->deserializeSimulation(data); },
                _simController));
        }
        delete progress;
        return false;
	}

	initSimulation(_simController->getContext()->getSymbolTable(), _simController->getContext()->getSimulationParameters());
	_view->refresh();

    delete progress;
    return true;
}

void MainController::onRecreateUniverse(SimulationConfig const& config, bool extrapolateContent)
{
    auto const recreateFunction = [&](Serializer* serializer) {
        recreateSimulation(serializer->retrieveSerializedSimulation());
    };
    _worker->add(boost::make_shared<_ExecuteLaterFunc>(recreateFunction));

    auto data = ModelGpuData(config->cudaConstants);

    Serializer::Settings settings{ config->universeSize, data.getData(), extrapolateContent };
    _serializer->serialize(_simController, static_cast<int>(ModelComputationType::Gpu), settings);
}

void MainController::onUpdateSimulationParameters(SimulationParameters const& parameters)
{
    auto progress = MessageHelper::createProgressDialog("Updating simulation parameters...", _view);

	_simController->getContext()->setSimulationParameters(parameters);

    QApplicationHelper::processEventsForMilliSec(500);
    delete progress;
}

void MainController::onUpdateExecutionParameters(ExecutionParameters const & parameters)
{
    auto progress = MessageHelper::createProgressDialog("Updating execution parameters...", _view);

    _simController->getContext()->setExecutionParameters(parameters);

    QApplicationHelper::processEventsForMilliSec(500);
    delete progress;
}

void MainController::onRestrictTPS(optional<int> const& tps)
{
	_simController->setRestrictTimestepsPerSecond(tps);
}

void MainController::onAddMostFrequentClusterToSimulation()
{
    _dataAnalyzer->addMostFrequenceClusterRepresentantToSimulation();
}

int MainController::getTimestep() const
{
    if (_simController) {
        return _simController->getContext()->getTimestep();
    }
    return 0;
}

SimulationConfig MainController::getSimulationConfig() const
{
	auto context = _simController->getContext();

	if (dynamic_cast<SimulationControllerGpu*>(_simController)) {
        auto data = ModelGpuData(context->getSpecificData());
        auto result = boost::make_shared<_SimulationConfig>();
        result->cudaConstants = data.getCudaConstants();
        result->universeSize = context->getSpaceProperties()->getSize();
		result->symbolTable = context->getSymbolTable();
		result->parameters = context->getSimulationParameters();
		return result;
	}
	else {
		THROW_NOT_IMPLEMENTED();
	}
}

SimulationMonitor * MainController::getSimulationMonitor() const
{
    return _simMonitor;
}

void MainController::connectSimController() const
{
	connect(_simController, &SimulationController::nextTimestepCalculated, [this]() {
		_view->getInfoController()->increaseTimestep();
	});
}

void MainController::addRandomEnergy(double amount)
{
	double maxEnergyPerCell = _simController->getContext()->getSimulationParameters().cellMinEnergy;
	_repository->addRandomParticles(amount, maxEnergyPerCell);
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::ActionController
	}, UpdateDescription::All);

}

